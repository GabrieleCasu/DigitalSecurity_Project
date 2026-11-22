from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

device = 'cpu'

# Initialize the models
mtcnn = MTCNN(keep_all=True, device=device)  # for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# load and prepare the reference image
ref_img = Image.open('images/probe/Impostor_N12/Probe12_FP10.jpg').convert('RGB')
ref_faces = mtcnn(ref_img)

if ref_faces is None or len(ref_faces) == 0:
    print("No faces detected in the reference image.")
    exit()

# use the first face and extract embedding
ref_face = ref_faces[0].unsqueeze(0).to(device)
with torch.no_grad():
    ref_embedding = resnet(ref_face)
# normalize the reference embedding
ref_embedding = ref_embedding.cpu().numpy().flatten()
ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)
# load comparison image
img = Image.open('images/probe/Impostor_N12/Probe12_FP1.jpg').convert('RGB')
boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
faces = mtcnn(img)

# test and visualization
plt.figure(figsize=(10, 8))
plt.imshow(img)
if boxes is not None and faces is not None:
    for i, (box, face2) in enumerate(zip(boxes, faces)):
        face = face2.unsqueeze(0).to(device)
        with torch.no_grad():
         emb = resnet(face)

        # normalize
        emb = emb.cpu().numpy().flatten()
        emb = emb / np.linalg.norm(emb)
        cosine_similarity = np.dot(ref_embedding, emb)
        threshold = 0.341 # optimal threshold, with only 1 image for identity it was 0.356
        label = "MATCH" if cosine_similarity > threshold else "NO MATCH"
        color = 'green' if label == "MATCH" else 'red'

        # generate a rectangle around the face
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          edgecolor=color, facecolor='none', linewidth=2))
        plt.text(x1, y1 - 10, f'{label} ({cosine_similarity:.3f})', color=color,
                 fontsize=10, backgroundcolor='white')
        # landmarks
        if landmarks is not None:
            for point in landmarks[i]:
                plt.plot(point[0], point[1], 'o', markersize=2, color='cyan')
        print(f"Face {i + 1}: Cosine similarity = {cosine_similarity:.3f} - {label}")
else:
    print("No faces detected in the comparison image.")

plt.title('Face comparison with MTCNN + FaceNet (cosine similarity)')
plt.axis('off')
plt.show()