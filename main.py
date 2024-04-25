import os
import openai
from config import OPENAI_API_KEY

class AI_Agent:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        openai.api_key = self.api_key
        self.initialized = False
    
    def generate_text(self, messages):
       try:
           response = openai.chat.completions.create(
               model="gpt-3.5-turbo",
               messages=messages
           )
           content = response.choices[0].message.content
           return content
       except Exception as e:
           print(f"An error occurred: {e}")
           return "Error in generating text."


    def generate_text_with_lore(self, user_input):
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": self.lore},
            {"role": "user", "content": user_input}
        ]
        return self.generate_text(messages)

    def initialize(self, task_description, instructions, lore, sample_dialogue):
        self.task_description = task_description
        self.instructions = instructions
        self.lore = lore
        self.sample_dialogue = sample_dialogue
        self.initialized = True
        print("Initialization complete.")

    def start_conversation(self):
        if not self.initialized:
            print("Please initialize the agent first.")
            return
        print(self.task_description)
        print(self.instructions)
        print()
        while True:
            user_input = input("Your response (type 'done' to finish): ")
            if user_input.lower() == "done":
                break
            response = self.generate_text_with_lore(user_input)
            print("Agent:", response)
            print()

def load_lore(file_path="lore.txt"):
    if not os.path.exists(file_path):
        print("Warning: Lore file not found.")
        return ""
    with open(file_path, "r") as file:
        lore = file.read()
    return lore

def prompt_user_for_prompts():
    print("Enter conversation prompts below. Type 'done' when finished.")
    prompts = []
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "done":
            break
        prompts.append(prompt)
    return prompts

def load_sample_dialogue(file_path="sample_dialogue.txt"):
    with open(file_path, "r") as file:
        dialogue = file.read()
    return dialogue

def load_instructions(file_path="instructions.txt"):
    if not os.path.exists(file_path):
        print("Warning: Instructions file not found.")
        return ""
    with open(file_path, "r") as file:
        instructions = file.read()
    return instructions

def main():
    agent = AI_Agent()

    while True:
        print("1. Initialize characters with your custom world")
        print("2. Start new conversation with character")
        print("3. Exit")

        choice = input("Select an option: ")

        if choice == "1":
            task_description = "Welcome to the Conversation Generator for RPG adventures!"
            instructions = load_instructions()
            lore = load_lore()
            sample_dialogue = load_sample_dialogue()
            agent.initialize(task_description, instructions, lore, sample_dialogue)

        elif choice == "2":
            if not agent.initialized:
                print("Please initialize the conversation first.")
            else:
                print("Starting a new conversation. Type 'done' at any time to end the conversation.")
                agent.start_conversation()

        elif choice == "3":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()