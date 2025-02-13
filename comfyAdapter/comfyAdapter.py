import websocket 
import uuid
import json
import urllib.request
import urllib.parse
import random
import os
import requests
from requests_toolbelt import MultipartEncoder

server_address = "141.76.64.123:8188"
client_id = str(uuid.uuid4())

workflow_file = "workflowSimple2.json"
input_image = "input/man-smoking.png"

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images


def upload_image(input_path, name, server_address, image_type="input", overwrite=False):
  with open(input_path, 'rb') as file:
    multipart_data = MultipartEncoder(
      fields= {
        'image': (name, file, 'image/png'),
        'type': image_type,
        'overwrite': str(overwrite).lower()
      }
    )

    data = multipart_data
    headers = { 'Content-Type': multipart_data.content_type }
    request = urllib.request.Request("http://{}/upload/image".format(server_address), data=data, headers=headers)
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read())
        path = data["name"]
        return path


def run_comfy_workflow(file):
    # upload an image
    comfyui_upload_image_path = upload_image(file,file,server_address)


    # load workflow
    with open(workflow_file, "r", encoding="utf-8") as wf_json:
        workflow = wf_json.read()


    prompt = json.loads(workflow)

    # set the input image
    prompt["33"]["inputs"]["image"] = comfyui_upload_image_path
    #prompt["44"]["inputs"]["image"] = comfyui_upload_image_path # old

    print(f"Running workflow: {workflow_file}")
    print(f"- with input image: {input_image}")

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt)
    ws.close() 

    # write generated image into result folder 
    outout_folder_name = "output"
    os.makedirs(outout_folder_name, exist_ok=True)

    out = []
    for node_id in images:
        for image_data in images[node_id]:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))
            result_image = f"{outout_folder_name}/image_{node_id}.png" 
            image.save(result_image, format="PNG")
            print(f"Result image saved: {result_image}")
            #image.show()
            out.append(result_image)
    return out

