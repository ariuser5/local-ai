from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load the model and tokenizer
    # model_name = "gpt2"  # Replace with another model if desired
    model_name = "microsoft/DialoGPT-medium"  # Replace with another model if desired
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

	# Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Input loop for the user
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Tokenize and generate a response
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        attention_mask = inputs.ne(tokenizer.pad_token_id).long()
        outputs = model.generate(
            inputs, 
            attention_mask=attention_mask, 
            max_length=50, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Lower randomness
            top_k=50,  # Focus on top 50 tokens
            top_p=0.9,  # Use nucleus sampling
            repetition_penalty=1.5  # Penalize repeated tokens
        )

        # Decode and print the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"LLM: {response}")

if __name__ == "__main__":
    main()