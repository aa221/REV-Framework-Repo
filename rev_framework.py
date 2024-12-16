import os
import pandas as pd
from together import Together
from openai import OpenAI
from typing import Optional
import re
from datetime import datetime
import multiprocessing
## retrieves the data set. 
human_eval_code = pd.read_parquet("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")

human_eval_code.head()

"""### Adding open ai key for GPT 4o"""

api_key_openai = "ENTER KEY HERE"

"""### Adding Together API key for QWEN"""

os.environ["TOGETHER_API_KEY"] = "Enter KEY API HERE" # Replace with your actual API key
client = Together()

"""### REV FRAMEWORK
#### QWEN as generator, GPT 4o as Verifier and Reasoner.
"""


class CodeGenerator:
    def __init__(self, api_key_openai: str, api_key_replicate):
        self.verifier = OpenAI(api_key=api_key_openai)
        self.formatter = OpenAI(api_key=api_key_openai)
        self.generator = client


    def clean_code_response(self, code: str) -> str:
        """Remove only the initial ```python and the final ``` from the code."""
        # Remove leading '```python\n'
        code = re.sub(r'^```python\n', '', code)
        # Remove trailing '```'
        code = re.sub(r'\n?```$', '', code)
        # Remove any leading/trailing whitespace
        return code.strip()



    def generate_code(self, prompt: str, entry_point: str,
                      feedback: Optional[str] = None,
                      previous_response: Optional[str] = None) -> str:
        # Construct the structured prompt based on whether feedback is available
        if feedback and previous_response:
            structured_prompt = (
                f"Original requirements:\n{prompt}\n\n"
                f"Previous implementation that failed:\n{previous_response}\n\n"
                f"Test feedback/issues:\n{feedback}\n\n"
                f"Please provide an improved Python function named exactly '{entry_point}' that addresses these issues."
                " Ensure it handles all edge cases and includes proper error handling."
                " Return only the code without any markdown formatting or explanations."
            )
        else:
            structured_prompt = (
                f"Requirements:\n{prompt}\n\n"
                f"Please generate a Python function named exactly '{entry_point}' that meets these requirements."
                " Ensure it handles all edge cases and includes proper error handling."
                " Return only the code without any markdown formatting or explanations."
            )


        # Prepare the input dictionary
        input_data = {
            "prompt": structured_prompt,  # User-defined structured prompt
        }

        # Prepare the messages list
        messages = [
            {"role": "user", "content": input_data["prompt"]},  # Include the structured prompt
        ]
        print(messages)

        # Make the API call
        output = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=messages,  # Pass the messages list
            max_tokens=512,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|im_end|>"],
            stream=False
        )


        output = output.choices[0].message.content

        response_text = output

        # Use clean_code_response to clean the generator's output
        code_to_execute = response_text

        # If code_to_execute is empty, use the entire response_text
        if not code_to_execute:
            code_to_execute = response_text.strip()

        # You can skip the formatter if not needed, or ensure it also outputs clean code
        # If you still use the formatter, clean its output as well
        response = self.formatter.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": (
                    "Format the following into executable code that can be run by an IDE. Remove any of the ``` and python "
                    "Do not return anything else:\n" + code_to_execute)}
            ],
            temperature=0,
            max_tokens=1000
        )

        # Clean the formatter's response
        code = response.choices[0].message.content
        code = self.clean_code_response(code)

        print("THE CODE")
        print(code)

        return code


    def verify_code(self, generated_code: str, prompt: str) -> tuple[bool, str]:
        system_prompt = """You are a thorough code reviewer. Analyze the code for:
1. Exact match with requirements
2. Edge case handling
3. Error handling
4. Potential test case failures
5. Correct function naming

If the code looks correct, respond with 'APPROVED'. Otherwise, list specific issues that need to be fixed."""

        response = self.verifier.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"Requirements:\n{prompt}\n\n"
                    f"Implementation:\n{generated_code}"
                )}
            ],
            temperature=0,
            max_tokens=1000
        )

        feedback = self.clean_code_response(response.choices[0].message.content)
        return "APPROVED" in feedback or "CORRECT" in feedback, feedback

def run_test(generated_code: str, test_code: str) -> tuple[bool, str]:
    # Create a new isolated namespace for testing
    test_namespace = {}

    try:
        # Execute both the test code and generated code
        exec(test_code, test_namespace)
        exec(generated_code, test_namespace)
    except Exception as e:
        return False, f"Code execution error: {str(e)}"

    # Extract function name
    function_match = re.search(r'def (\w+)\(', generated_code)
    if not function_match:
        return False, "Function definition not found"

    function_name = function_match.group(1)
    if function_name not in test_namespace:
        return False, f"Function '{function_name}' not properly defined"

    # Run tests
    try:
        test_namespace['check'](test_namespace[function_name])
        return True, "All tests passed"
    except AssertionError as e:
        return False, f"Test assertion failed: {str(e)}"
    except Exception as e:
        return False, f"Test execution error: {str(e)}"

def generate_and_verify(prompt: str, entry_point: str, test_code: str, api_key_openai: str, api_key_replicate, max_attempts: int = 3) -> str:
    generator = CodeGenerator(api_key_openai, api_key_replicate)
    previous_response = None
    previous_feedback = None

    for attempt in range(max_attempts):
        # Generate code
        generated_code = generator.generate_code(
            prompt=prompt,
            entry_point=entry_point,
            feedback=previous_feedback,
            previous_response=previous_response
        )

        # Patch the code to address common errors
        if "resul" in generated_code:
            generated_code = generated_code.replace("resul", "result")

        # Run actual tests first
        test_passed, test_feedback = run_test(generated_code, test_code)
        if test_passed:
            return generated_code

        # Get detailed verification feedback
        #_, verification_feedback = generator.verify_code(generated_code, prompt)

        # Combine test and verification feedback
        combined_feedback = f"Test feedback: TEST(S) FAILED \nVerification feedback: {None}"
        print(f"Attempt {attempt + 1} feedback:\n{combined_feedback}")

        # Update for next iteration
        previous_response = generated_code
        previous_feedback = combined_feedback

        # If we're on the last attempt and still haven't succeeded
        if attempt == max_attempts - 1:
            raise Exception(f"Failed to generate correct code after {max_attempts} attempts. Last feedback: {combined_feedback}")

    return generated_code


def per_problem(index, results, human_eval_code, api_key_openai, api_key_replicate):
    try:
        # Extract test case data for this index
        prompt = human_eval_code["prompt"].iloc[index]
        entry_point = human_eval_code["entry_point"].iloc[index]
        test_code = human_eval_code["test"].iloc[index]

        # Attempt to generate and verify code
        try:
            generated_code = generate_and_verify(
                prompt=prompt,
                entry_point=entry_point,
                test_code=test_code,
                api_key_openai=api_key_openai,
                api_key_replicate=api_key_replicate,
                max_attempts=1  # Adjustable number of attempts
            )
            results['passed'] += 1
            print(f"✓ Test {index} passed")
        except Exception as e:
            # Handle cases where generation and verification fail
            results['failed'] += 1
            results['failed_indices'].append(index)
            results['error_logs'][index] = str(e)
            print(f"✗ Test {index} failed: {str(e)}")

    except Exception as e:
        # Handle unexpected errors
        results['failed'] += 1
        results['failed_indices'].append(index)
        results['error_logs'][index] = str(e)
        print(f"✗ Test {index} failed with error: {str(e)}")


def testing_framework(human_eval_code, api_key_openai: str, api_key_replicate: str):
    start_time = datetime.now()
    manager = multiprocessing.Manager()
    results = manager.dict({
        'passed': 0,
        'failed': 0,
        'failed_indices': manager.list(),
        'error_logs': manager.dict()
    })

    total_tests = len(human_eval_code)

    for index in range(total_tests):
        print(f"\nTesting {index}/{total_tests} ({(index/total_tests)*100:.1f}%)")

        p = multiprocessing.Process(target=per_problem, args=(
            index, results, human_eval_code, api_key_openai, api_key_replicate))
        p.start()
        p.join(120)  # Wait up to 2 minutes

        if p.is_alive():
            print(f"Time limit exceeded for test {index}, moving to next problem.")
            p.terminate()
            p.join()
            results['failed'] += 1
            results['failed_indices'].append(index)
            results['error_logs'][index] = "Time limit exceeded"

    # Convert managed objects to regular data structures
    results = {
        'passed': results['passed'],
        'failed': results['failed'],
        'failed_indices': list(results['failed_indices']),
        'error_logs': dict(results['error_logs'])
    }

    # Calculate and print summary
    duration = datetime.now() - start_time
    pass_rate = (results['passed'] / total_tests) * 100

    print("\n=== Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass rate: {pass_rate:.2f}%")
    print(f"Duration: {duration}")
    print("\nFailed test indices:", results['failed_indices'])

    return results

# Example usage:
#results = testing_framework(human_eval_code, api_key_openai, api_key_replicate)

class CodeReasoner(CodeGenerator):
    def reason_and_plan(self, prompt: str) -> str:
        """Generate a reasoning plan to solve the problem based on the given prompt."""
        system_prompt = """You are a coding assistant focused on logical reasoning.
        For the given problem, create a step-by-step plan to solve it.
        The plan should include:
        1. An understanding of the problem.
        2. Logical steps to solve the problem.
        3. Potential edge cases to consider.
        Return the plan only without explanations."""

        response = self.verifier.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem:\n{prompt}"}
            ],
            temperature=0.5,
            max_tokens=500
        )

        plan = self.clean_code_response(response.choices[0].message.content)
        print("\nGenerated Plan:")
        print(plan)
        return plan

    def generate_code_with_reasoning(self, prompt: str, entry_point: str, test_code: str, max_attempts: int = 3) -> str:
        """Generate code using reasoning, planning, and feedback."""
        previous_response = None
        previous_feedback = None

        for attempt in range(max_attempts):
            # Generate a plan first
            reasoning_plan = self.reason_and_plan(prompt)

            # Combine the plan with the original prompt for the generator
            enhanced_prompt = (
                f"Problem:\n{prompt}\n\n"
                f"Plan:\n{reasoning_plan}\n\n"
                f"Please generate a Python function named exactly '{entry_point}' "
                f"that implements the above plan, handles edge cases, and includes error handling."
                " Return only the code without any markdown formatting or explanations."
            )

            # Generate code
            generated_code = self.generate_code(
                prompt=enhanced_prompt,
                entry_point=entry_point,
                feedback=previous_feedback,
                previous_response=previous_response
            )

            # Run tests
            test_passed, test_feedback = run_test(generated_code, test_code)
            if test_passed:
                print("\n✓ Code passed the tests.")
                return generated_code

            # Feedback failed code back to the reasoning layer
            feedback_prompt = (
                f"Problem:\n{prompt}\n\n"
                f"Plan:\n{reasoning_plan}\n\n"
                f"Generated Code:\n{generated_code}\n\n"
                f"Test Feedback:\n{test_feedback}\n\n"
                "Based on the feedback, update the plan and suggest a revised implementation."
            )

            revision_plan = self.reason_and_plan(feedback_prompt)
            print(f"Attempt {attempt + 1} failed. Revising plan:\n{revision_plan}")

            # Update the prompt and responses for the next iteration
            previous_response = generated_code
            previous_feedback = test_feedback

            if attempt == max_attempts - 1:
                raise Exception(f"Failed to generate correct code after {max_attempts} attempts. Feedback:\n{test_feedback}")

        return generated_code

# Modified testing framework to use the reasoner
def testing_framework_with_reasoning(human_eval_code, api_key_openai: str, api_key_replicate: str):
    start_time = datetime.now()
    manager = multiprocessing.Manager()
    results = manager.dict({
        'passed': 0,
        'failed': 0,
        'failed_indices': manager.list(),
        'error_logs': manager.dict()
    })

    reasoner = CodeReasoner(api_key_openai, api_key_replicate)
    total_tests = len(human_eval_code)

    for index in range(total_tests):
        print(f"\nTesting {index}/{total_tests} ({(index/total_tests)*100:.1f}%)")

        prompt = human_eval_code["prompt"].iloc[index]
        entry_point = human_eval_code["entry_point"].iloc[index]
        test_code = human_eval_code["test"].iloc[index]

        try:
            # Generate and verify code with reasoning
            generated_code = reasoner.generate_code_with_reasoning(
                prompt=prompt,
                entry_point=entry_point,
                test_code=test_code,
                max_attempts=3  # Adjustable number of attempts
            )
            results['passed'] += 1
            print(f"✓ Test {index} passed")
        except Exception as e:
            results['failed'] += 1
            results['failed_indices'].append(index)
            results['error_logs'][index] = str(e)
            print(f"✗ Test {index} failed: {str(e)}")

    # Convert managed objects to regular data structures
    results = {
        'passed': results['passed'],
        'failed': results['failed'],
        'failed_indices': list(results['failed_indices']),
        'error_logs': dict(results['error_logs'])
    }

    # Calculate and print summary
    duration = datetime.now() - start_time
    pass_rate = (results['passed'] / total_tests) * 100

    print("\n=== Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass rate: {pass_rate:.2f}%")
    print(f"Duration: {duration}")
    print("\nFailed test indices:", results['failed_indices'])

    return results

# Example usage:
if __name__ == '__main__':
    results = testing_framework_with_reasoning(human_eval_code, api_key_openai, api_key_replicate="None")