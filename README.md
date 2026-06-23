# Overview
A Complete Timeline Editor For LTX 2.3. This is the sucessor of my previous nodes, and has loads of features in it. It was originally based off of [Kijai's Prompt Relay node](https://github.com/kijai/ComfyUI-PromptRelay) and LTX Sequencer/Multi Image Loader nodes.


This will be a collection of free resources for ComfyUI.

Hopefully it will make creating cool stuff easier.

All of my nodes are created with the help of AI, so there may or may not be redundant, messy code.

**❗❗IMPORTANT❗❗**


This is a Modded LTX director node with some extra options to help you create videos with references sheets

This node uses Ollama Locally : you will need to install it (very lightweight) and 

- In your Ollama model folder run this command to install qwen 3.5 2b q4 (1.9gb) "ollama run huihui_ai/qwen3.5-abliterated:2B"

- The model won't eat memory while generating since there is an auto clear VRAM when you hit Run or after 5min.

- You can still enter your description manually if you don't want to install it but it works very well!

- Both references mode work (fixed), Licon MSR and Ghost Mask.

- Use their Lora, that is an important step : https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference/tree/main

- Enjoy!

## ▶️ YouTube Tutorial Videos

<a href="https://youtu.be/j28z5PZXkKk?si=TPeWKm2BW-uA3BAx">
<img width="642" height="718" alt="Capture d&#39;écran 2026-06-17 234136" src="https://github.com/user-attachments/assets/89433110-8b78-4971-8098-3a9e9226f460" />


<table>
  <tr>
    <td>
      <p align="center">LTX Director Trailer</p>
      <a href="https://www.youtube.com/watch?v=fZgtkRcu4_k">
        <img src="https://img.youtube.com/vi/fZgtkRcu4_k/0.jpg" alt="LTX Director Trailer" width="400">
      </a>
    </td>
    <td>
      <p align="center">LTX Director Tutorial</p>
      <a href="https://www.youtube.com/watch?v=vM60pJJqqEI">
        <img src="https://img.youtube.com/vi/vM60pJJqqEI/0.jpg" alt="LTX Director Tutorial" width="400">
      </a>
    </td>
  </tr>
</table>

## ❓ How to install nodes

- Navigate to your `/ComfyUI/custom_nodes/ folder`
- Delete your old "WhatDreamsCost-ComfyUI" Folder
- Run `git clone -b main_cs https://github.com/CGlideWhatDreamsCost-CSGlide
.git`




<img width="596" height="668" alt="Capture d&#39;écran 2026-06-13 234036" src="https://github.com/user-attachments/assets/2dfe7992-a72a-4eda-a103-41b3a8f77714" />


# 🔄 Recent Updates

**v1.3.3**
  * **LTX Director Hotfix 2**
    - Fixed duration_seconds input issue.
    - Made both duration widgets visible at all times now
    - Implemented audio latent fix to improve compatibility


**v1.3.2**
  * **LTX Director Hotfix**
    - Fixed epsilon input overlapping custom_width input
    - Fixed invisible widgets in nodes 2.0 when toggling widget visibility through settings menu

If anyone finds anymore bugs or has idea for improvements please let me know! 


**v1.3.1**
  * **LTX Director Example Workflow Fix**
    - Minor fix to the example workflow (i forgot to set the clip loader type to ltxv lol)
    
 **v1.3.0**
  * **New nodes: LTX Director and LTX Director Guide**
    - A complete timeline editor that can do almost everything. It's my most ambitious node so far and the successor to LTX Sequencer/Multi Image Loader.

 **v1.2.9**
  * **Fixed every known issue with Multi Image Loader and added text output to Speech Length Calculator**
  
    - Removed the completely useless drag and drop animations (now it's snappy and no longer finicky)
    - Fixed the node resizing on nodes 2.0 
    - Updated grid logic to fit images better
    - Added ablity to right click images to copy/open/save images
    - Fixed the "invisible hitbox" underneath node issue (actually this time).

  Also added a text output to the Speech Length Calculator node (can't believe i didn't do this initially)

<details>
  <summary>Click to view older Updates</summary>

 **v1.2.8**
  * **Updated Load Video UI and Color Conversion**
    * Added crop mode, a simple interface to crop videos. It also include various aspect ratio presets.
    * Updated color conversion to ensure colors are as accurate as possible. Will first check metadata for colorspace, and if metadata is missing then it will guess the colorspace based on video dimensions.
    * Updated display mode toggle UI to be more understandable 

 **v1.2.7**
  * **New Node: Load Video UI**

Custom Node to Trim, Resize, and Preview Videos in Realtime
  
   **v1.2.6**
  * **Updated Speech Length Calculator UI**

Also added duration output to the Load Audio UI node

 **v1.2.5**
  * **Updated Load Audio UI Node**
    * Added Duration Setting
    * Made the whole selection bar draggable
    * Fixed Trimmed UI to show centiseconds
    
 **v1.2.4**
 * **New Node: Load Audio UI**

Overhaul of the load audio node. Features a simple interface to easily trim audio. Also allows dragging and dropping files (fixes the original node that doesn't allow dropping in videos). Also compatible with nodes 2.0.

 **v1.2.3**
  * **Workflow Update + Minor Bug Fix** 
    * Added new workflow that is compatible with the latest ComfyUI version (as of 4/27/26). The new workflow also included an option to include custom audio, and has minor improvements of the previous workflows.
    * Fixed minor bug with Multi Image Loader that blocked mouse input in a small area under the node 🤷‍♂️

**v1.2.0**
  * **New Node: Speech Length Calculator** 
  
  Automatically output in realtime how long a video should be based on the dialouge. 

**v1.1.0**
  * Added resize_method to the Multi Image Loader node for more resize options
  * Added insert_mode which allows you to enter in seconds instead of frames on the LTX Sequencer node
  * Updated workflows with more notes
  * Re-added tiny vae to workflows
  * Fixed various bugs
  * more things i can't rememeber
  
**This update will change the node layouts, so be sure to update your workflows or else they won't work properly.**

❗❗❗ **New Tutorial on using these nodes available: https://www.youtube.com/watch?v=aXDIr8eNovI**  ❗❗❗
</details>

# ⚙️ Custom Nodes


## LTX Director
<img width="1481" height="833" alt="Clipboard Image (2)" src="https://github.com/user-attachments/assets/08f3fe53-9393-4f5d-9de5-58b229fbed47" />



**Main Features:**
- **Fully Functional Timeline Editor:** I spent hours studying various video editors and ended up with this design. If anyone has ideas for improvements let me know! I will adding documentation on all the functions soon.
- **Prompt Relay integrated:** This unlocks the ability to have granular control over video generation. For more information on Prompt Relay go here, https://gordonchen19.github.io/Prompt-Relay/
- **First, Middle, Last Frame Support:** This has by far the easiest method of creating first/last frames videos. It supports any number of keyframes, and will be the successor of my previous nodes.
- **Custom Audio Support:** Import, trim, and combine your own audio clips in this node. Enabling custom audio is as simple as clicking 1 button. It is also compatible with every other feature in the node, include first/last frames, t2v, i2v, and prompt relay.
- **Image to Video:** Part of the goal of this node was to make it easier to do everything, including Image to Video. It has built in resize functionality, and of course all the benifits of the prompt relay and custom audio integration.
- **Text to Video:** Use text segments to create T2V videos. Compatible with all other features of the node.

Download workflows here: https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI/tree/main/example_workflows

**Tutorial videos and documentation coming soon**




For those asking, I coded all of this myself (LOL). 
