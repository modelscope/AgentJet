import ast
import re
from typing import Any, Callable, Dict, List, Tuple


# Abstract Syntax Tree Visitor to extract variable names
class AstStructureExtractor(ast.NodeVisitor):
    """Visitor pattern to extract all keys (variable names)"""

    def __init__(self):
        self.keys = set()
        # Define builtin function list to avoid dependency on different behaviors of __builtins__
        self.builtin_names = {
            "min",
            "max",
            "abs",
            "round",
            "int",
            "float",
            "sum",
            "len",
            "str",
            "bool",
            "list",
            "dict",
            "tuple",
            "set",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "all",
            "any",
            "bin",
            "hex",
            "oct",
            "chr",
            "ord",
            "pow",
            "divmod",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "delattr",
            "callable",
            "iter",
            "next",
            # Add other potential builtin functions as needed
        }

    def visit_Name(self, node):
        # Collect all variable names, excluding builtin functions
        if node.id not in self.builtin_names:
            self.keys.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Handle attribute access like "astuner.rollout.max_env_worker"
        # Reconstruct the full attribute path
        full_key = self._get_full_attribute_name(node)
        if full_key and not self._is_builtin_attribute(full_key):
            self.keys.add(full_key)
        # Don't call generic_visit to avoid duplicate processing of child nodes

    def _get_full_attribute_name(self, node):
        """Recursively get the full attribute name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_full_attribute_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

    def _is_builtin_attribute(self, attr_name):
        """Check if it's an attribute of a builtin module (like math.sin)"""
        # Can extend this list as needed
        builtin_modules = {"math", "os", "sys", "json", "re", "datetime"}
        parts = attr_name.split(".")
        return len(parts) > 1 and parts[0] in builtin_modules


def split_keys_and_operators(
    operation_str: str, preserved_field: List[str] = []
) -> Tuple[List[str], Callable[[Dict[str, Any]], Any]]:
    """
    Parse expression string using AST and extract keys and operators

    Input example: (min(astuner.rollout.max_env_worker // astuner.rollout.n_vllm_engine, 64))
    Output example: (['astuner.rollout.max_env_worker', 'astuner.rollout.n_vllm_engine'], <function for computing result>)
    """

    # Parse the expression
    print(operation_str)
    try:
        tree = ast.parse(operation_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Expression syntax error: {operation_str}") from e

    # use Abstract Syntax Tree to extract all keys
    extractor = AstStructureExtractor()
    extractor.visit(tree)
    keys = sorted(list(extractor.keys))

    # Create evaluation function
    def eval_func(values: Dict[str, Any]) -> Any:
        # Check if all required keys exist
        missing_keys = [key for key in keys if key not in values]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        # Create mapping from key names to safe variable names
        key_mapping = {}
        safe_expression = operation_str

        # Sort by key length in descending order to replace longer keys first
        sorted_keys = sorted(keys, key=len, reverse=True)

        for i, key in enumerate(sorted_keys):
            # Create a safe variable name for each key
            safe_var = f"var_{i}"
            key_mapping[safe_var] = values[key]

            # Use regex to precisely match and replace key names
            # Ensure no partial matching (e.g., won't match "a.b.c" in "a.b.cd")
            pattern = re.escape(key) + r"(?![a-zA-Z0-9_.])"
            safe_expression = re.sub(pattern, safe_var, safe_expression)

        # Create a safe namespace for evaluation
        namespace = {
            "__builtins__": {
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "int": int,
                "float": float,
                "sum": sum,
                "len": len,
                # Can add more safe builtin functions as needed
            }
        }

        # Add mapped variables to namespace
        namespace.update(key_mapping)

        # Evaluate the expression
        try:
            result = eval(safe_expression, namespace)
            return result
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression '{operation_str}': {e}") from e

    print(f"Extracted keys: {keys}")
    return keys, eval_func


# Test examples
if __name__ == "__main__":
    # Example 1
    operation_str1 = "(astuner.data.train_batch_size * astuner.rollout.num_repeat * astuner.rollout.multi_turn.expected_steps)"
    known_operators1 = []

    keys1, func1 = split_keys_and_operators(operation_str1)
    print("Example 1:")
    print(f"Extracted keys: {keys1}")

    values1 = {
        "astuner.data.train_batch_size": 32,
        "astuner.rollout.num_repeat": 4,
        "astuner.rollout.multi_turn.expected_steps": 10,
    }
    result1 = func1(values1)
    print(f"Computed result: {result1}")  # 32 * 4 * 10 = 1280
    print()

    # Example 2
    operation_str2 = "(astuner.rollout.max_env_worker // astuner.rollout.n_vllm_engine)"
    known_operators2 = []

    keys2, func2 = split_keys_and_operators(operation_str2)
    print("Example 2:")
    print(f"Extracted keys: {keys2}")

    values2 = {"astuner.rollout.max_env_worker": 100, "astuner.rollout.n_vllm_engine": 8}
    result2 = func2(values2)
    print(f"Computed result: {result2}")  # 100 // 8 = 12
    print()

    # Example 3: Mixed operators
    operation_str3 = "(a * b / c + d - e)"
    known_operators3 = []

    keys3, func3 = split_keys_and_operators(operation_str3)
    print("Example 3:")
    print(f"Extracted keys: {keys3}")

    values3 = {"a": 100, "b": 5, "c": 10, "d": 20, "e": 5}
    result3 = func3(values3)
    print(f"Computed result: {result3}")  # 100 * 5 / 10 + 20 - 5 = 65.0

    # Example 4
    operation_str4 = "(min(astuner.rollout.max_env_worker // astuner.rollout.n_vllm_engine, 64))"
    known_operators4 = []

    keys4, func4 = split_keys_and_operators(operation_str4)
    print("Example 4:")
    print(f"Extracted keys: {keys4}")

    values4 = {
        "astuner.rollout.max_env_worker": 512,
        "astuner.rollout.n_vllm_engine": 4,
    }
    result4 = func4(values4)
    print(f"Computed result: {result4}")  # 64
