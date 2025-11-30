import re
from typing import Any, Callable, Dict, List, Tuple


def split_keys_and_operators(
    operation_str: str, known_operators: List[str]
) -> Tuple[List[str], Callable[[Dict[str, Any]], Any]]:
    """
    Split keys and operators from an operation expression string and return a callable function

    Args:
        operation_str: Operation expression string, e.g., "(astuner.data.train_batch_size * astuner.rollout.num_repeat)"
        known_operators: List of known operators, e.g., ["*", "//", "/"]

    Returns:
        Tuple[List[str], Callable]:
            - List of extracted keys
            - Callable function that accepts a dictionary and returns the computed result
    """
    # Remove leading/trailing parentheses and whitespace
    cleaned_str = operation_str.strip().strip("()")

    # Sort operators by length in descending order to avoid "//" being matched by "/" first
    sorted_operators = sorted(known_operators, key=len, reverse=True)

    # Build regex pattern, escape special characters
    escaped_operators = [re.escape(op) for op in sorted_operators]
    pattern = "|".join(escaped_operators)

    # Split string while preserving separators
    parts = re.split(f"({pattern})", cleaned_str)

    # Extract keys (parts that are not operators)
    keys = []
    for part in parts:
        stripped_part = part.strip()
        if stripped_part and stripped_part not in known_operators:
            keys.append(stripped_part)

    # # Operator mapping
    # operator_map = {
    #     "+": operator.add,
    #     "-": operator.sub,
    #     "*": operator.mul,
    #     "/": operator.truediv,
    #     "//": operator.floordiv,
    #     "%": operator.mod,
    #     "**": operator.pow,
    # }

    def compute_function(values_dict: Dict[str, Any]) -> Any:
        """
        Compute the expression result based on the provided values dictionary

        Args:
            values_dict: Dictionary of key-value pairs

        Returns:
            Computed result
        """
        # Build expression string by replacing keys with actual values
        expr = cleaned_str

        # Sort keys by length in descending order to avoid incorrect replacement
        # when shorter key names are contained in longer ones
        sorted_keys = sorted(keys, key=len, reverse=True)

        for key in sorted_keys:
            if key not in values_dict:
                raise ValueError(f"Missing key in values_dict: {key}")
            # Use word boundary to ensure complete match
            expr = re.sub(rf"\b{re.escape(key)}\b", str(values_dict[key]), expr)

        # Safely evaluate the expression
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return result
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {expr}. Error: {e}")

    return keys, compute_function


# Test examples
if __name__ == "__main__":
    # Example 1
    operation_str1 = "(astuner.data.train_batch_size * astuner.rollout.num_repeat * astuner.rollout.multi_turn.expected_steps)"
    known_operators1 = ["*", "//", "/"]

    keys1, func1 = split_keys_and_operators(operation_str1, known_operators1)
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
    known_operators2 = ["*", "//", "/"]

    keys2, func2 = split_keys_and_operators(operation_str2, known_operators2)
    print("Example 2:")
    print(f"Extracted keys: {keys2}")

    values2 = {"astuner.rollout.max_env_worker": 100, "astuner.rollout.n_vllm_engine": 8}
    result2 = func2(values2)
    print(f"Computed result: {result2}")  # 100 // 8 = 12
    print()

    # Example 3: Mixed operators
    operation_str3 = "(a * b / c + d - e)"
    known_operators3 = ["*", "//", "/", "+", "-"]

    keys3, func3 = split_keys_and_operators(operation_str3, known_operators3)
    print("Example 3:")
    print(f"Extracted keys: {keys3}")

    values3 = {"a": 100, "b": 5, "c": 10, "d": 20, "e": 5}
    result3 = func3(values3)
    print(f"Computed result: {result3}")  # 100 * 5 / 10 + 20 - 5 = 65.0
