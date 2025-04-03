from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from utils import get_device, extract_action, ascii_to_image
from huggingface_hub import login
from breakout import breakout
from openai import OpenAI 
from google import genai

import numpy as np
import argparse
import imageio
import torch
import time
import csv
import os


# Disable tokenizer parallelism warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Breakout ASCII')
    parser.add_argument('--model', default='gpt4o',
                        help='What model would you like to use?  (gpt4o, geminipro, geminiflash, llama)')
    args = parser.parse_args()
    
    api_value = args.model 

    if api_value == "gpt4o":
        print("\nSetting up GPT-4o...\n")
        with open("OPENAI_API_KEY.txt", "r") as f:
            openai_api_key = f.read().strip()
        client = OpenAI(api_key=openai_api_key)

    elif api_value == "geminiflash":
        print("\nSetting up Gemini Flash 2.0...\n")
        with open("GOOGLE_API_KEY.txt", "r") as f:
            google_api_key = f.read().strip()
        # Initialize the Google GenAI client using the new SDK.
        client = genai.Client(api_key=google_api_key)
        # Create a chat session with Gemini 2.0 Flash.
        chat = client.chats.create(model="gemini-2.0-flash")

    elif api_value == "geminipro":
        print("\nSetting up Gemini Pro 2.5...\n")
        with open("GOOGLE_API_KEY.txt", "r") as f:
            google_api_key = f.read().strip()
        # Initialize the Google GenAI client using the new SDK.
        client = genai.Client(api_key=google_api_key)
        # Create a chat session with Gemini 2.0 Flash.
        chat = client.chats.create(model="gemini-2.5-pro-exp-03-25")

    elif api_value == "llama":
        print("\nSetting up Llama 3.2 3B...\n")
        with open("HG_API_KEY.txt", "r") as f:
            hugging_face_api_key = f.read().strip()
        login(token=hugging_face_api_key)
        set_seed(42)
        device = get_device()
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError("Invalid selection. Please enter 1, 2, or 3.")

    print("\nModel set up complete...\n")

    with open("system_prompt.txt") as prompt:
        systemprompt = prompt.read().strip()
    
    messages = [{"role": "system", "content": systemprompt}]
    
    env = breakout(render_mode="human")
    obs, _ = env.reset(seed=42)
    frames = [ascii_to_image(obs, num_cols=env.WIDTH, num_rows=env.LENGTH)]
    total_rewards = 0
    cumulative_rewards = []
    action_list = []
    max_message_len = 9  # Limit conversation length

    # Game loop: 1000 steps.
    for i in range(1000):
        action = None
        # Provide up to 3 chances for a correct action.
        for _ in range(3):
            messages.append({
                    "role": "user",
                    "content": 
                            f"This is the current game state:\n{obs}\n\n"
                            "Based on the provided game state, please determine the best action to take. Remember:\n"
                            "- Provide your detailed reasoning between <reasoning> and </reasoning> meta tags.\n"
                            "- Then, immediately provide your chosen action between <action> and </action> meta tags, with no extra text or formatting."
                            "Remember your action output should only be one of these 3 options <action> 0 </action> for NOOP, <action> 1 </action> for LEFT and <action> 2 </action> for RIGHT\n"
                            "Show me what the current game state is:\n"
                        })
            
            if api_value == "gpt4o":
                response = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=messages,
                    temperature=0,
                )
                output_decode = response.choices[0].message.content
            elif api_value == "geminiflash" or api_value == "geminipro":
                # For Gemini Flash, use the chat session.
                response = chat.send_message(message=messages[-1]["content"])
                output_decode = response.text
            elif api_value == "llama":
                input_ids = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", return_dict=True
                ).to(get_device())
                outputs = model.generate(**input_ids, max_new_tokens=4096)
                output_decode = tokenizer.decode(outputs[0])
            
            with open('./all_responses.txt', "a") as file:
                file.write(str(output_decode) + '\n\n')

            try:
                action = extract_action(output_decode)
                print(f"Extracted action: {action}")
                break
            except ValueError as e:
                print("Extraction error:", e)
                messages.append({
                    "role": "user",
                    "content": "You did not put the action between the <action> and </action> meta tags. PROVIDE THE ENDING ACTION ONLY BETWEEN THESE META TAGS."
                })

        messages.append({"role": "assistant", "content": output_decode})
        action_list.append(action)
        
        obs, reward, done, truncated, info = env.step(action)
        total_rewards += reward
        
        cumulative_rewards.append(total_rewards)
        
        frames.append(ascii_to_image(obs, num_cols=env.WIDTH, num_rows=env.LENGTH))
        
        if len(messages) >= max_message_len:
            messages.pop(1)
            messages.pop(1)
        
        if done:
            obs, _ = env.reset(seed=42)
            frames.append(ascii_to_image(obs))
            done = False

        print("Step ", i)
        time.sleep(0.1)

    print("\n\n Total Reward: ", total_rewards)
    print("Saving video of performance...")
    video_filename = "breakout.mp4"
    images = [np.array(frame) for frame in frames if frame is not None]
    imageio.mimwrite(video_filename, images, fps=env.metadata["render_fps"])
    print(f"Saved video as {video_filename}")

    print("Saving actions and cumulative rewards...")
    header = ["actions", "cumulative_rewards"]
    with open('./actions_rewards.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for action, cum_reward in zip(action_list, cumulative_rewards):
            writer.writerow([action, cum_reward])
    
    print("\nTest complete, Thank you!\n")