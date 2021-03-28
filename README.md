# CrowdTally
State-of-the-art human detection for crowd counting

## What it does
CrowdTally consists of two components: crowd counting from static images (either uploaded by the user or shared by other users) and crowd counting from videos. 
1. **Crowd counting for images**: Users can upload their own images of the beaches and share it with the community. They can also visualize the crowd count from images that others have uploaded (which are stored in Firebase). Once the image is uploaded, the human detection AI model will output the number of beachgoers and generate an annotated image showing bounding boxes for all the detections.
2. **Crowd counting for videos**: Users can upload a video footage for the AI model to analyze and annotate in real time. As video detection is more computationally expensive, users have a choice between two models: (1) a slower model with **higher precision** (IterDet), and (2) a **lightweight** but less accurate model (YOLO) depending on their usage requirements.

## How we built it
CrowdTally uses a **state-of-the-art human detection model** IterDet (ranked first of all object detection models on the WiderPerson annotation challenge with the highest mean average precision) to analyze images and videos of crowds to generate a real-time tally of the crowd count along with the corresponding bounding boxes showing the position of each detected person. CrowdTally also contains a second model YOLO that is more lightweight but nevertheless still highly accurate which the user can choose to use if inference speed is critical.
We **fine-tuned** IterDet on crowd images using **PyTorch**. We then coded the front-end using **Streamlit** and hosted the website using ngrok.
If users upload their own images to the web application, they can either keep their images private (the image will not be stored and will only be accessible by the user) or to share their images with the community which will be stored in **Firebase Cloud Storage**. When users click on the "Search for crowdsourced images" button, the web application queries the Firebase Cloud Storage to retrieve the images other users have uploaded.

## Challenges we ran into
Both models are computationally expensive so we spent a substantial amount of time fine-tuning them. This meant we weren't sure how well the models would perform on unseen images until near the end of the hackathon when the models finally finished training. We're glad both perform really well, in fact even better than we expected (we're especially impressed by how well the models can accurately detect humans even when they appear so small in the image)!

## Accomplishments that we're proud of
We are extremely proud to have trained and fine-tuned state-of-the-art machine learning models in such a short span of time and to have also created a front-end that successfully links to the backend that we were able to successfully deploy!
Being the first time we used Firebase Storage, we learnt a lot about NoSQL databases and witnessed their power and flexibility.

## What's next for CrowdTally
Depending on the needs of the Citizen Science project, we are happy to continue refining CrowdTally and introducing more features to help scientists conduct research. 
We are interested in research because we truly believe in the potential of research to revolutionize the way we lead our lives. Many societal advancements are attained via research and innovation, and we believe technology can serve as a key driving force. All of us can contribute to research even if we are not scientists - the crowdsourcing aspect of the beach tally app shows us precisely this. This is the beauty of research, and we're excited and honored to be a part of this effort!
