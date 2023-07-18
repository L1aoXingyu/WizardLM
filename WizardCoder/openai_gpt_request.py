import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')

def send_prompt(system_prompt, user_prompt):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    # usage
    response = send_prompt("You are a helpful assistant.", "Tell me a joke.")
    print(response)
