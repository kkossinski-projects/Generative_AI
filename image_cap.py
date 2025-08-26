from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

def main():
    # Load the pretrained processor and model
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Path to your image file - make sure this file exists in your project directory
    img_path = "cofficient_team.jpeg"
    
    try:
        # Load and convert the image to RGB
        image = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found. Please upload it to your project directory.")
        return

    # Prepare inputs for the model (no question needed for captioning)
    text = "the image of"
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Print the caption
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()
