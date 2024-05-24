# Floor PLan Use Case
## Objective:
To detect and retrieve area from the provided image using Google Vision API and u-net-based computer vision model. The user uploads two floor-plan(images) to compare in the user interface as below:
![UI1](https://github.com/vasant1012/floorplan/assets/52622703/2db6c43d-b232-41c4-9e9f-9e908d99c0f4)

## Getting Started
Both plans will be sent to API to get ocr result and post-processing of the result will provide the final table which has features, dimensions and its respective areas.
![UI2](https://github.com/vasant1012/floorplan/assets/52622703/911bdf59-4190-468e-b92f-4566c53dedd4)

Also Simultaneously it will send to deep learning model to predict quadrant-wise area. The model takes the image, breaks it into 4 quadrants, and predicts multiple classes as features of the floorplan. Those classes will be mapped to features generated from api and also area is calculated using pixel information from the image. It will displayed as:
![UI3](https://github.com/vasant1012/floorplan/assets/52622703/93ac59ba-64e0-4085-b985-f9af8b504747)

Finally, after we get complete information click on the "Summary" button to generate a summary based on area. The summary also has a conclusion that compares feature-wise area and suggests the best floorplan to the user.
![UI4](https://github.com/vasant1012/floorplan/assets/52622703/a11c23e8-b0df-4f04-ae24-e0f43e01eea0)

## Tech Stack Involved
- *Google Vision API*: to get information from floorplan images uploaded by the user in the User interface in terms of OCR and post-process result into a Table that has room, dimension, and area.
- *U-net-based CV model*: Deep learning model that predicts the area in terms of pixels. Then it will be mapped with generated data from API and get the quadrants-wise area of uploaded plans.
- *Plotly Dash UI*: The user interface is a medium to map the generated data from API and model prediction output for final summary generation.

*This is the base version. 1.0*
