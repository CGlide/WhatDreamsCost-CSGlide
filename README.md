
A Complete Timeline Editor For LTX 2.3. This is the sucessor of my previous nodes, and has loads of features in it. It was originally based off of [Kijai's Prompt Relay node](https://github.com/kijai/ComfyUI-PromptRelay) and my LTX Sequencer/Multi Image Loader nodes.

https://github.com/user-attachments/assets/68dc826f-c65f-4f1e-86cd-4f2df17bacd8



# Overview

Director CS Node is a Modded version from "WhatdreamsCost"

## ▶️ YouTube Tutorial Videos



## ❓ How to install nodes

- Navigate to your `/ComfyUI/custom_nodes/ folder`
- Run `git clone -b main_cs https://github.com/CGlide/WhatDreamsCost-CSGlide.git`


## What's new in 0.23

### MSR Prefix

New dropdown in the settings menu. `17 / 25 / 33 / 41 / 49 / 57 / 65`, default is 41.

This is the length of the reference runway — the little slideshow of your reference images that runs before the actual video, so the model has time to look at them and lock the identity. Longer runway, stronger lock.

And to be clear because I got this wrong myself at first: this has nothing to do with how many references you use. You still get 3 `@ref` slots, always. It's a bigger stage, not a bigger cast.

**Careful:** 49 and up only work with Licon MSR **V2**. V1 was not trained on those lengths. And the longer you go, the more memory it eats — if you're already close to OOM on a long generation, don't push this to 65 and expect miracles.

### Prompt Relay ON / OFF

The Prompt Relay toggle now actually changes the whole UI, not just the backend.

**Relay ON** — what you had before. One prompt per segment, each one lands on its own moment in the timeline. Best when you want fine control over what happens and when.

**Relay OFF (guide mode)** — the segment prompt box disappears completely and the Global Prompt box grows to fill the space. Now you write one prompt for the whole thing and your images do the driving. The label even changes to *Global Prompt (IC-LoRA)* so you know where you are.

Honestly I use OFF more than I expected. It's more predictable graphically — you're not fighting three prompts pulling the shot in different directions. Some people will tell you it's faster too, and technically yes, but the sampler is basically all of your generation time so don't switch for that. Switch because the result is cleaner.

Your global prompt box height is remembered, by the way. Toggle back to ON and it goes back to how you had it.

### Prompt Zones + zone dots

Turn on Prompt Zones and you get the coloured ribbon on the timeline showing where each prompt applies.

New in this one: little coloured dots right after the **SEGMENT PROMPT** label, one per zone, same colours as the ribbon. Click a dot and it jumps to that segment and loads its prompt. The selected one gets a white outline so you can see where you are.

Small thing. I use it constantly now. Only shows up when Prompt Zones is ON and relay is ON, because otherwise there are no zones to point at.

### Convert to Text Segment

Right-click an image segment → **Convert to Text Segment**. It drops the image but **keeps your prompt**.

Before this I was deleting the segment and retyping the prompt like an idiot. Sits right under *Convert to Image Anchor* in the menu.

### @ref works for anything now

The tags used to be `@char1 / @char2 / @char3` and the describer assumed everything was a person. So if you loaded a car it would try to tell you about its hair.

Now it's `@ref1 / @ref2 / @ref3` and the analyze prompt figures out on its own whether it's looking at a character or an object — vehicle, prop, creature, whatever — and describes it the right way. Reference sheets of a jeep work exactly like reference sheets of a person.

Old `@char1` and `@character1` still work. I didn't break your old timelines.

### 50 fps and the MSR warning

50 fps is in the frame rate presets now. And when MSR is on and you're **not** at 50, a small orange **⚠ MSR 50 recommended** appears next to the frame rate. Click it and it sets 50 for you.

Licon MSR is trained at 50. I tried 48 thinking it's close enough — the motion goes doubled and jittery. It's not close enough. Use 50.

### Timeline handling

- Aspect ratio lock between width and height. Change one, the other follows.
- Middle mouse to drag the timeline around, and to zoom.
- New resolution presets.

---

## Settings menu reference

Click the gear on the node.

| Setting | What it does |
| --- | --- |
| **Save / Save As / Load Timeline** | Timelines are files. Save the good ones. |
| **Hide / Show Widgets** | Collapses the raw node widgets when you're working from the timeline UI. |
| **Prompt Relay** | ON = one prompt per segment. OFF = global prompt only, images act as guides. |
| **MSR Prefix** | Reference runway length. 17–65 frames, default 41. 49+ needs MSR V2. |
| **Display Mode** | Frames or Seconds on the ruler. |
| **Show Filenames** | Filename overlay on image segments. |
| **Prompt Zones** | The coloured prompt ribbon and the zone dots. |
| **Epsilon** | How sharply a prompt is confined to its segment. Lower = tighter. Leave it alone unless you know why you're touching it. |
| **Divisible By** | Rounds your resolution so the model doesn't complain. |
| **Img Compression** | Compression on the images you load in. |
| **Workspace Folder** | Opens the folder where timelines and assets live. |
| **Provider / Base URL / Model** | The backend that writes your reference descriptions when you hit Analyze. |

---

## Careful with this

**ComfyUI Manager "Update All" does a hard reset.** If you edited any file in this folder, it's gone. No warning. Happened to me. Commit or back up first.

**Don't press "Sync fork" on GitHub** if you forked this. It pulls the original repo over the top of yours. I did that once and it deleted a pull request I had open. 🤦

**MSR is not the everyday tool.** Ghost Mask is what I reach for most of the time — the motion is more natural. MSR is the specialist: when you need the face or the object to match your reference image exactly, that's when you switch. Don't leave it on for everything.
