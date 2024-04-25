import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import os

class AI_Agent:
    def __init__(self, model_name='gpt2'):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = TFGPT2LMHeadModel.from_pretrained(model_name)
        self.task_description = ""
        self.instructions = ""
        self.lore = ""
        self.prompts = []

    def initialize(self, task_description, instructions, lore, sample_dialogue):
        self.task_description = task_description
        self.instructions = instructions
        self.lore = lore
        self.sample_dialogue = sample_dialogue
        

    def warm_up_with_sample(self):
        # Encode and process the sample dialogue to 'warm-up' the model
        if self.sample_dialogue:
            input_ids = self.tokenizer.encode(self.sample_dialogue, return_tensors="tf")
            # Simply doing a forward pass to let the model 'think' through the sample dialogue
            self.model(input_ids, training=False)

        
    def generate_text(self, prompt, max_length=250, temperature=1.0, top_k=50, top_p=0.95):
        input_ids = self.tokenizer.encode(prompt, return_tensors="tf")
        attention_mask = tf.ones(input_ids.shape, dtype=tf.int32)  # Ensure all tokens are attended to
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id  # Set pad token
        )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

        
    def generate_text_with_lore(self, prompt):
        prompt_with_lore = self.lore + " " + prompt
        return self.generate_text(prompt_with_lore)

    def start_conversation(self):
        print(self.task_description)
        print(self.instructions)
        print()
        for prompt in self.prompts:
            print(prompt)
            user_input = input("Your response: ")
            response = self.generate_text_with_lore(user_input)
            response_without_lore = response[len(self.lore):].strip()
            print("Agent:", response_without_lore)
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
    # Specify the model name
    model_name = "gpt2"

    # Initialize AI agent
    agent = AI_Agent(model_name)

    while True:
        print("1. Initialize characters with your custom world")
        print("2. Start new conversation with character")
        print("3. Exit")

        choice = input("Select an option: ")

        if choice == "1":
            # Initialize conversation
            task_description = "Welcome to the Conversation Generator for RPG adventures!"
            instructions = load_instructions()
            lore = load_lore()
            sample_dialogue = load_sample_dialogue()
            agent.initialize(task_description, instructions, lore, sample_dialogue)
            # Optionally add a warm-up step here if you want to 'process' the lore/sample
            agent.warm_up_with_sample()  
            print("Initialization complete.")

        elif choice == "2":
            # Start new conversation
            prompt = input("Enter your starting prompt: ")
            response = agent.generate_text(prompt)
            print("AI:", response)

        elif choice == "3":
            # Exit
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
