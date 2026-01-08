from elevenlabs.conversational_ai.conversation import ClientTools
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
import openai
import requests
from PIL import Image
from io import BytesIO
import base64

def searchWeb(parameters):
    query = parameters.get("query")
    results = DuckDuckGoSearchRun(query=query)
    return results

def save_to_txt(parameters):
    filename = parameters.get("filename")
    data = parameters.get("data")

    formatted_data = f"{data}"

    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_data + "\n")

def create_html_file(parameters):
    file_name = parameters.get("filename")
    data = parameters.get("data")
    title = parameters.get("title")

    formatted_html = f""" 
    <!DOCTYPE html>
    <html lang="en">
    <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>{title}</title>
    </head>
    <body>
        <h1>{title}</h1>
        <div>{data}</div>
    </body>
    </html>
    """
    
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(formatted_html)

def generate_image(parameters):
    """
    Generate image using Stability AI (much cheaper than OpenAI!)
    Cost: ~$0.003-0.004 per image vs OpenAI's $0.04
    """
    prompt = parameters.get("prompt")
    filename = parameters.get("filename")
    size = parameters.get("size", "1024x1024")  # Supports: 1024x1024, 1024x768, 768x1024, etc.
    save_dir = parameters.get("save_dir", "generated_images")

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure filename has .png extension
    filename = filename if filename.endswith(".png") else filename + ".png"
    filepath = os.path.join(save_dir, filename)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("stability_api_key")
    
    if not api_key:
        raise ValueError("stability_api_key not found in .env file!")
    
    # Parse size
    try:
        width, height = map(int, size.split("x"))
    except:
        width, height = 1024, 1024  # Default size
    
    # Stability AI API endpoint
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    body = {
        "text_prompts": [
            {
                "text": prompt,
                "weight": 1
            }
        ],
        "cfg_scale": 7,
        "height": height,
        "width": width,
        "samples": 1,
        "steps": 30,
    }
    
    print(f"🎨 Generating image with Stability AI...")
    print(f"📝 Prompt: {prompt}")
    print(f"📐 Size: {width}x{height}")
    
    try:
        # Make API request
        response = requests.post(url, headers=headers, json=body)
        
        if response.status_code != 200:
            error_msg = f"Error {response.status_code}: {response.text}"
            print(f"❌ {error_msg}")
            
            # Helpful error messages
            if response.status_code == 401:
                raise Exception("Invalid API key. Check STABILITY_API_KEY in .env file!")
            elif response.status_code == 402:
                raise Exception("Insufficient credits. Add credits at platform.stability.ai/account/billing")
            else:
                raise Exception(error_msg)
        
        # Parse response and save image
        data = response.json()
        
        for artifact in data.get("artifacts", []):
            if artifact.get("finishReason") == "SUCCESS":
                # Decode base64 image
                image_data = base64.b64decode(artifact["base64"])
                
                # Save image
                image = Image.open(BytesIO(image_data))
                image.save(filepath)
                
                print(f"✅ Image saved to: {filepath}")
                
                return {
                    "success": True,
                    "filepath": filepath,
                    "size": f"{width}x{height}",
                    "message": f"Image generated successfully at {filepath}"
                }
        
        raise Exception("No successful image generated")
        
    except Exception as e:
        print(f"❌ Error generating image: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to generate image: {str(e)}"
        }


# Register tools
client_tools = ClientTools()
client_tools.register("searchWeb", searchWeb)
client_tools.register("saveToTxt", save_to_txt)
client_tools.register("createHtmlFile", create_html_file)
client_tools.register("generateImage", generate_image)