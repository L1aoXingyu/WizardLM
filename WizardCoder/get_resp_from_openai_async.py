import argparse
import asyncio
import uuid
import aiohttp
from pydantic import BaseSettings
import json
import os
import signal
from asyncio import Queue

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='WizardCoder/code_instruct_p2.json')
parser.add_argument('--output_name', type=str, default='code_evlo_20k')
parser.add_argument('--output_variant', type=str, default='solution_p2')
parser.add_argument('--save_dir', type=str, default='evolved_instruction')
parser.add_argument('--num_retries', type=int, default=50)
parser.add_argument('--pause_between_retries', type=int, default=5)
parser.add_argument('--max_concurrent_requests', type=int, default=50)
args = parser.parse_args()

# network request parameters
num_retries = args.num_retries
pause_between_retries = args.pause_between_retries # seconds
max_concurrent_requests = args.max_concurrent_requests # max number of concurrent requests

# dataset parameters
output_name = args.output_name
output_variant = args.output_variant
data_dir = args.save_dir

os.makedirs(data_dir, exist_ok=True)

# Configuration
config = {
    "processed_items_file_prefix": f'{data_dir}/{output_name}_',
    "processed_items_file_postfix": f'{output_variant}_',
    "responses_file_prefix": f'{data_dir}/{output_name}_',
    "responses_file_postfix": f'{output_variant}_',
}

class Settings(BaseSettings):
    LLM_SERVER_ADDRESS = "localhost"
    LLM_SERVER_PORT = "8080"

    XMIND_ROOT_URL = "https://cognitive.xmindai.com/api/"
    XMIND_API_KEY = ""
    CYBER_KYUSHU_ROOT_URL = "http://47.254.87.225/xmind-ai/chat/completions"


settings = Settings()

model = "gpt-3.5-turbo-0613"
n_ctx = 4096

def text_prompt_to_chat_prompt(prompt: str, role: str = "system"):
    assert isinstance(prompt, str), f"Expected a text prompt, got {prompt}"
    return [
        {"role": role, "content": prompt},
    ]


def check_cyber_kyushu_response_error_code(resultCode=-1):
    return resultCode in [9000, 9001, 9002, 9991, 9999]


# Function to load data from JSON file
def load_json_file(file_path):
    try:
        if file_path.endswith("json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.endswith("jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f.readlines()]
        print(f'{file_path} len: {len(data)}')
        return data
    except FileNotFoundError:
        return []

items = load_json_file(args.input_path)

# Load processed items if any
processed_items = load_json_file(
    config["processed_items_file_prefix"]+config["processed_items_file_postfix"]+'processed_items.json')

# Load responses if any
responses = load_json_file(
    config["responses_file_prefix"]+config["responses_file_postfix"]+'responses.json')


async def write_to_file(queue):
    while True:
        response, item = await queue.get()  # Wait until a response is available

        # If we receive the sentinel value, exit the loop
        if item is None:
            queue.task_done()
            break

        # append response to responses
        responses.append(response)

        # Save responses with indentation
        with open(f'{config["responses_file_prefix"]}{config["responses_file_postfix"]}responses.json', 'w') as f:
            json.dump(responses, f, indent=4)

        # append item to processed_items and save it
        processed_items.append(item)
        # Save processed items with indentation
        with open(f'{config["processed_items_file_prefix"]}{config["processed_items_file_postfix"]}processed_items.json', 'w') as f:
            json.dump(processed_items, f, indent=4)

        queue.task_done()  # Indicate that a formerly enqueued task is complete.


async def send_evolve_request(session, item, queue, semaphore):
    if item in processed_items:
        return


    for i in range(num_retries):
        await semaphore.acquire()  # Manually acquire the lock
        try:
            url = settings.CYBER_KYUSHU_ROOT_URL
            headers = {
                "X-Token": f"{os.environ.get('CYBER_KYUSHU_API_KEY')}",
                "Content-Type": "application/json",
                "Request-Origion": "Knife4j",
                "Accept": "*/*",
            }

            payload = {
            "messages": text_prompt_to_chat_prompt(item, role="user"),
            "model": model,
            "uuid": str(uuid.uuid4()),
        }
            for j in range(len(payload["messages"])):
                context = payload["messages"][j]["content"]
            if len(context) > n_ctx:
                payload["messages"][j]["content"] = context[:n_ctx]

            async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
                response.raise_for_status()
                data = await response.json()
                response_content = data["choices"][0]["message"]["content"]
                # Add response and item to queue
                await queue.put((response_content, item))
            break
        except aiohttp.ClientError as e:
            semaphore.release()  # Release the lock before sleeping
            print(f"Request failed for {item} with error {e}, attempt {i+1}")
            await asyncio.sleep(pause_between_retries)
        finally:
            semaphore.release()  # Release the lock if not retrying


async def shutdown(signal, loop):
    """Cleanup tasks tied to the service's shutdown."""
    print(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]

    print(f"Cancelling {len(tasks)} tasks")
    for task in tasks:
        print(f"Cancelling {task}")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()



async def main():
    # register signals for interruption
    loop = asyncio.get_running_loop()
    for s in {signal.SIGTERM, signal.SIGINT}:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

    queue = Queue()

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(send_evolve_request(session, item, queue, semaphore)) for item in items]
        # create write_to_file task
        file_writer = asyncio.create_task(write_to_file(queue))
        await asyncio.gather(*tasks, return_exceptions=True)
        await queue.join()  # Wait for all items in the queue to be processed.
        await queue.put((None, None))
        await file_writer


# Run the event loop
asyncio.run(main())
