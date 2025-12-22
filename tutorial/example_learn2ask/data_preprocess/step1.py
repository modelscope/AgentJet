import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_info_extraction import LLM_info_extraction, parse_llm_output
from message_splitter import split_session_to_json_lines


def process_jsonl_file(
    input_file, output_file, model_call_mode="online_api", max_retries=3, max_workers=16, **kwargs
):
    """
    Process all sessions in a JSONL file and save results to output file using multi-threading.
    Supports resuming from previous progress if interrupted.

    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
        model_call_mode (str): Either "online_api" or "local_vllm"
        max_retries (int): Maximum number of retries for LLM calls
        max_workers (int): Maximum number of threads for parallel processing
        **kwargs: Additional parameters for API calls

    Returns:
        str: Success message or error information
    """
    progress_file = output_file + ".progress"
    
    def load_progress():
        """Load progress from progress file. Returns set of completed line numbers."""
        if os.path.exists(progress_file):
            with open(progress_file, "r", encoding="utf-8") as f:
                return set(int(line.strip()) for line in f if line.strip())
        return set()
    
    def process_single_session(args):
        """Worker function to process a single session."""
        line_num, line = args
        if not line.strip():
            return line_num, None, None
        try:
            session = json.loads(line)
            print(
                f"Processing session {session.get('session_id', 'unknown')} (line {line_num})..."
            )
            processed_lines = process_session(
                session, model_call_mode, max_retries, **kwargs
            )
            return line_num, processed_lines, None
        except json.JSONDecodeError as e:
            return line_num, None, f"Warning: Skipping invalid JSON at line {line_num}: {e}"
        except Exception as e:
            return line_num, None, f"Warning: Error processing session at line {line_num}: {e}"
    
    try:
        # Load previous progress
        completed_lines = load_progress()
        if completed_lines:
            print(f"Resuming from previous progress. {len(completed_lines)} lines already completed.")
        
        # Read all lines first
        with open(input_file, "r", encoding="utf-8") as infile:
            all_lines = list(enumerate(infile, 1))
        
        total_lines = len(all_lines)
        # Filter out already completed lines
        lines_to_process = [(num, line) for num, line in all_lines if num not in completed_lines]
        
        if not lines_to_process:
            print("All lines already processed.")
            # Clean up progress file
            if os.path.exists(progress_file):
                os.remove(progress_file)
            return f"All lines already processed. Results in {output_file}"
        
        print(f"Processing {len(lines_to_process)} remaining lines out of {total_lines} total.")
        
        # State for ordered writing
        results_buffer = {}  # line_num -> processed_lines
        next_line_to_write = min(num for num, _ in lines_to_process)
        write_lock = threading.Lock()
        progress_lock = threading.Lock()
        
        # Open output file in append mode if resuming, otherwise write mode
        file_mode = "a" if completed_lines else "w"
        outfile = open(output_file, file_mode, encoding="utf-8")
        progress_out = open(progress_file, "a", encoding="utf-8")
        
        def flush_buffer():
            """Write all consecutive completed results from buffer to file."""
            nonlocal next_line_to_write
            while next_line_to_write in results_buffer:
                processed_lines = results_buffer.pop(next_line_to_write)
                if processed_lines:
                    for processed_line in processed_lines:
                        outfile.write(processed_line + "\n")
                outfile.flush()
                # Save progress
                with progress_lock:
                    progress_out.write(f"{next_line_to_write}\n")
                    progress_out.flush()
                next_line_to_write += 1
                # Skip lines that were already completed or empty
                while next_line_to_write <= total_lines and next_line_to_write not in dict(lines_to_process):
                    next_line_to_write += 1
        
        try:
            # Process sessions in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_session, item): item[0] for item in lines_to_process}
                
                for future in as_completed(futures):
                    line_num, processed_lines, error = future.result()
                    if error:
                        print(error)
                    
                    with write_lock:
                        results_buffer[line_num] = processed_lines
                        flush_buffer()
        finally:
            outfile.close()
            progress_out.close()
        
        # Clean up progress file on successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        return f"Successfully processed. Results saved to {output_file}"

    except Exception as e:
        return f"Error processing JSONL file: {str(e)}"


def process_session(session, model_call_mode="online_api", max_retries=3, **kwargs):
    """
    Pipeline function that splits messages into rounds and extracts info from each round's remaining chat.

    Args:
        session (dict): Session dictionary containing 'session_id', 'diagn', and 'messages' keys
        model_call_mode (str): Either "online_api" or "local_vllm"
        max_retries (int): Maximum number of retries for LLM calls
        **kwargs: Additional parameters for API calls

    Returns:
        list: List of JSON strings with added "info_set" key, or error information
    """
    # Step 1: Split messages into JSON lines
    json_lines = split_session_to_json_lines(session)

    # Step 2: Process each JSON line with LLM info extraction
    processed_lines = []

    for line in json_lines:
        data = json.loads(line)
        remaining_chat = data.get("remaining_chat", "")

        # Retry loop for LLM calls
        info_set = None
        for attempt in range(max_retries):
            try:
                # Call LLM info extraction (using mock function for testing)
                llm_response = LLM_info_extraction(remaining_chat, model_call_mode, **kwargs)

                info_set = parse_llm_output(llm_response)

                if isinstance(info_set, list):
                    break
                else:
                    # If parsing failed, this is an error message
                    print(f"Attempt {attempt + 1} failed: {info_set}")
                    if attempt < max_retries - 1:
                        time.sleep(24)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with exception: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(24)  # Shorter wait for testing
        
        if info_set is None:
            raise Exception(f"failed to generate {session}")
        data["info_set"] = info_set
        processed_lines.append(json.dumps(data, ensure_ascii=False))

    return processed_lines


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default="examples/learn_to_ask/data_raw/train_origin.jsonl"
    )
    parser.add_argument(
        "--output_file", type=str, default="examples/learn_to_ask/data_raw/train_processed.jsonl"
    )
    parser.add_argument(
        "--model_call_mode", type=str, choices=["online_api", "local_vllm"], default="online_api"
    )
    args = parser.parse_args()
    print(
        process_jsonl_file(
            input_file=args.input_file,
            output_file=args.output_file,
            model_call_mode=args.model_call_mode,
            # Additional parameters for API calls
        )
    )