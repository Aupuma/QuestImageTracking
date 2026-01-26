# Quest Image Tracking

This repository enables **single and multi-image detection and tracking** using OpenCV with the **Meta Quest 3/3S Passthrough Camera API**.  
It provides a robust solution for **augmented reality applications** on Quest devices, supporting both individual and multiple image targets.

For a demonstration, check out the following video:

[![thumbnailImageTracking (Personalizado)](https://github.com/user-attachments/assets/b94b261c-1704-4e83-8e03-0b41e320d883)](https://www.youtube.com/watch?v=VO-p3Q3inrI)

## Dependencies

⚠ **Important Notice**  
When opening the project for the first time, you will likely encounter errors. This is because **OpenCV for Unity** is not yet installed. **Please ignore the errors initially, proceed to open the project, and install OpenCV for Unity manually.**  

This project uses **OpenCV for Unity**.   
Please purchase and install it from the Unity Asset Store:  
[OpenCV for Unity](https://assetstore.unity.com/packages/tools/integration/opencv-for-unity-21088?locale=en-US)  

The original version of the project works with **OpenCV for Unity v2.6.5**
The QuestImageTracking_New project works with MetaXR SDK v83 and higher and **OpenCV for Unity v3+**. This version uses Tasks to avoid blocking the main thread so dynamic tracking is smoother

## 🔖 Marker Preparation  

1. Set **Read/Write** on your Image Texture to Enabled
2. Set Max Size to **512**
3. Create an **AR Tracked Image Data**: Right Click -> Create -> AR-> Tracked Image Data
4. Fill the data:
    - **id**: can be any arbitrary identifier
    - **Marker Texture**: your pattern texture (not all images will be valid, depends on the OpenCV pattern detector)
    - **Physical size**: this needs to match the real size of your target in meters, otherwise the depth/scale of the virtual object won't be correct

## Usage

**ImageTracking.cs** includes two tracking modes:
- **Dynamic**: tracks the targets continuously applying some filtering to reduce jitter
- **Static**: performs an initial detection and pose estimation until the target is stabilized. You can adjust in the settings how many poses will be used to consider the target stable.

To add a new tracked image to the system, add a new element in the **Tracked Images** array, including the Tracked Image Data and the object from the scene you want to show with the image. 

To make things easier you can use the **AR Tracked Object** prefab, which includes the **Image Tracking Helper** that can be used to align the elements to the target and the **AR Tracked Image Event** with an **OnImageStabilized** event that is fired in Static mode once the target is stabilized.

Two example scenes have been included in the project, each one showcasing each mode. You can build them and run in your headset to know the system works correctly.

### 🔄 Example scene controls  

- Press the **A button** to toggle debug mode. Debug mode will show the tracking helpers and a label with the active tracking mode.

- Press the **B button** to switch between tracking modes

## Current limitations and areas of improvement

- The pattern detection and pose estimation has a **notable performance impact** which adds on top of the Passthrough Camera Use. Decreasing **Processing Downsample Factor** will lead to a performance increase but a decrease of tracking quality.

- On Static mode pattern detection will stop once a target is stabilized to keep the best possible performance. Once the virtual object of the active pattern gets outside the camera frustum the pattern detection will resume. Currently this mode works with only one marker at a time

## Reference Repositories

This implementation is based on the following sample repositories:  

- [Unity-PassthroughCameraApiSamples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples)  
- [QuestCameraKit](https://github.com/xrdevrob/QuestCameraKit)  
- [Quest ArUco Marker Tracking](https://github.com/TakashiYoshinaga/QuestArUcoMarkerTracking)  

## Contact

If you have any questions, feel free to reach out:  

- **X (Twitter)**: [@aurepuerta_dev](https://x.com/aurepuerta_dev)  
- **LinkedIn**: [Aurelio Puerta Martín](www.linkedin.com/in/aurelio-puerta-martin)  
