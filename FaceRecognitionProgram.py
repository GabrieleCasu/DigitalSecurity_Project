import os #files and directories
import numpy as np  #math and array
import matplotlib.pyplot as plt  #graphichs
from facenet_pytorch import MTCNN, InceptionResnetV1 #normalize faces and extract embeddings
from PIL import Image #images
from sklearn.metrics import roc_curve, auc, det_curve  # calculate the performances
import pickle #save/load results
from tqdm import tqdm #to visualize the progress
import torch #helps the models

class FaceRecognition:
    #initialization phase
    def __init__(self, device='cpu'):
        self.device = device
        self.mtcnn = MTCNN(keep_all=False, device=device) #detect only the biggest face
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) #mode=evaluation, use pretrained dataset
        self.gallery_embeddings = {}
        self.probe_embeddings = {}
        self.similarity_scores = []
        self.labels = [] # 1=genuine, 0=impostor
        self.detailed_results = []

    #extract the embedding from the face in an image
    def extract_face_embedding(self,image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            face = self.mtcnn(img) #normalize the face
            if face is None:
                return None
            face = face.unsqueeze(0).to(self.device) #adds a dimension that consents the procedure
            with torch.no_grad(): #we don't need the gradient during evaluation
                embedding = self.resnet(face) #use the neural network, returns 512 numbers
            return embedding.cpu().numpy().flatten() #convert in a 1D vector from a tensor

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    #loading the gallery set
    def load_gallery_set(self,gallery_path):
        print(" \nLoading gallery set...")
        for img_file in tqdm(os.listdir(gallery_path)): #all the files
            if not img_file.lower().endswith('_gallery.jpg'): continue

            #extract the identity ID from the file name to do all the comparisons
            try:
                identity_id = img_file.split('_')[0] #takes the first part before the underscore
            except Exception as e:
                print(f"File name not found: {img_file}: {e}")
                continue
            if identity_id not in self.gallery_embeddings: #checks if is a new identity
                self.gallery_embeddings[identity_id] = [] #if he/she is, creates a new list
            img_path = os.path.join(gallery_path, img_file) #creates the path
            embedding = self.extract_face_embedding(img_path)
            if embedding is not None: #adds a dictionary to this identity
                self.gallery_embeddings[identity_id].append({
                    'embedding': embedding,
                    'filename': img_file,
                    'path': img_path
                })

        print(f"\nGallery successfully loaded: {len(self.gallery_embeddings)} identities")

    #loading the probe set
    def load_probe_set(self,probe_path):
        print("Loading probe set...")
        all_items = os.listdir(probe_path)
        folders = [item for item in all_items if os.path.isdir(os.path.join(probe_path,item))] #only the directories

        for identity_folder in tqdm(folders):
            identity_path = os.path.join(probe_path, identity_folder)
            identity_id = identity_folder
            if identity_folder.startswith('Identity_N'): #uniforms the names
               identity_id = identity_folder.replace('Identity_N','IdentityN')

            self.probe_embeddings[identity_id] = []
            #load all the images in the subdirectory
            probe_files = os.listdir(identity_path)

            for img_file in probe_files:
                if img_file.lower().endswith('.jpg'):
                    img_path = os.path.join(identity_path, img_file)
                    embedding = self.extract_face_embedding(img_path)
                    if embedding is not None:
                        self.probe_embeddings[identity_id].append({
                            'embedding': embedding,
                            'filename': img_file,
                            'path': img_path
                        })
                    else: print(f"Embedding error: {img_file}")
        print(f" Probe successfully loaded: {len(self.probe_embeddings)} identities")

    #calculating the similarity between two embeddings
    def calculate_similarity(self, emb1, emb2, metric='cosine'):
        emb1 = emb1/np.linalg.norm(emb1)
        emb2 = emb2/np.linalg.norm(emb2)
        #we divide by the length of the vector to get a vector of length 1, crucial for FaceNet
        if metric == 'cosine':
            #range [-1,1]
                return np.dot(emb1, emb2)
        #higher the result(similarity), more similar the identities

        elif metric == 'euclidean':
            #range [0,2]
                euclidean_distance = np.linalg.norm(emb1 - emb2)
                similarity = 2.0 - euclidean_distance #to invert green and red in the graphic and adapt to cosine
                return similarity

        elif metric == 'manhattan':
            #range [-30,0]
            manhattan_distance = np.sum(np.abs(emb1 - emb2))
            similarity = 2.0 - manhattan_distance #many negative values because the range [0,2vd]
            return similarity
    #different ranges have consequences with the threshold, but have more clear graphics

    #matching function, every probe for every gallery, Nprobe * Ngallery comparisons
    def perform_matching(self,similarity_metric='cosine'):
            print("Performing matching...")
            self.similarity_scores = []
            self.labels = []
            self.detailed_results = []
            total_comparisons = 0
            for probe_id in self.probe_embeddings:
                for gallery_id in self.gallery_embeddings:
                    total_comparisons += len(self.probe_embeddings[probe_id]) * len(self.gallery_embeddings[gallery_id])
            with tqdm(total=total_comparisons, desc='Matching') as pbar:
                # it returns pair (key,value) from the dictionary
                for probe_id, probe_list in self.probe_embeddings.items(): #every probe
                    for probe_data in probe_list: #every image of the identity for that probe
                        probe_emb = probe_data['embedding']
                        for gallery_id, gallery_list in self.gallery_embeddings.items(): #for every gallery
                            for gallery_data in gallery_list:
                                gallery_emb = gallery_data['embedding']

                                #calculate similarities
                                score = self.calculate_similarity(probe_emb, gallery_emb, similarity_metric)
                                is_genuine = 1 if probe_id == gallery_id else 0
                                self.similarity_scores.append(score)
                                self.labels.append(is_genuine)
                                #save the details
                                self.detailed_results.append({
                                    'probe_file': probe_data['filename'],
                                    'gallery_file': gallery_data['filename'],
                                    'score': score,
                                    'is_genuine': is_genuine
                                })

                                pbar.update(1) #updates the loading bar

            self.similarity_scores = np.array(self.similarity_scores)
            self.labels = np.array(self.labels)
            genuine_scores = self.similarity_scores[self.labels == 1] #takes only the true elements
            impostor_scores = self.similarity_scores[self.labels == 0]

            print(f"Matching completed:")
            print(f" Total comparisons: {len(self.similarity_scores)}")
            print(f" Total genuine: {len(genuine_scores)}")
            print(f" Total impostors: {len(impostor_scores)}")
            print(f" Medium genuine score: {genuine_scores.mean():.3f}")
            print(f" Medium impostor score: {impostor_scores.mean():.3f}")

    # graphic of the distributions of the scores (genuine/impostors)
    def plot_score_distributions(self):
        genuine_scores = self.similarity_scores[self.labels == 1]
        impostor_scores = self.similarity_scores[self.labels == 0]
        plt.figure(figsize=(10, 6))
        plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', color='red', density=True) #50 intervals, 70% transparence for the overallocation, probability value
        plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', color='green', density=True)
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Score Distributions - Genuine vs Impostor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # ROC curve
    def plot_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.similarity_scores) #calculates every point of the curve
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})') #quality metric
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

        return roc_auc

    # DET curve
    def plot_det_curve(self):
        fpr, fnr, thresholds = det_curve(self.labels, self.similarity_scores)
        plt.figure(figsize=(8, 6))
        plt.loglog(fpr, fnr, color='blue', lw=2)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('False Negative Rate (FNR)')
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.grid(True, alpha=0.3)
        plt.show()

    # FMR and FNMR with different thresholds
    def plot_fmr_fnmr(self):
        thresholds = np.linspace(self.similarity_scores.min(),self.similarity_scores.max(),100) #100 values from min to max score
        fmr_values = []
        fnmr_values = []
        for threshold in thresholds:
            predictions = (self.similarity_scores > threshold).astype(int) #array of t/f
            tp = np.sum((predictions == 1) & (self.labels == 1))  # True Positives
            fp = np.sum((predictions == 1) & (self.labels == 0))  # False Positives
            tn = np.sum((predictions == 0) & (self.labels == 0))  # True Negatives
            fn = np.sum((predictions == 0) & (self.labels == 1))  # False Negatives

            # FMR = FP / (FP + TN) - False Match Rate
            fmr = fp / (fp + tn) if (fp + tn) > 0 else 0
            # FNMR = FN / (FN + TP) - False Non-Match Rate
            fnmr = fn / (fn + tp) if (fn + tp) > 0 else 0

            fmr_values.append(fmr)
            fnmr_values.append(fnmr)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, fmr_values, label='FMR (False Match Rate)', color='red', lw=2)
        plt.plot(thresholds, fnmr_values, label='FNMR (False Non-Match Rate)', color='blue', lw=2)
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.title('FMR vs FNMR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # find the EER (Equal Error Rate)
        eer_idx = np.argmin(np.abs(np.array(fmr_values) - np.array(fnmr_values)))
        eer = (fmr_values[eer_idx] + fnmr_values[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        return eer, eer_threshold

    #CMS curve
    def plot_cms_curve(self,max_rank=15):
        #for every probe finds the rank of the correct match in the gallery
        ranks = []
        for probe_id in self.probe_embeddings:
            for probe_data in self.probe_embeddings[probe_id]:
                probe_emb = probe_data['embedding']
                gallery_scores = []
                gallery_ids = []
                for gallery_id in self.gallery_embeddings:
                    for gallery_data in self.gallery_embeddings[gallery_id]:
                        gallery_emb = gallery_data['embedding']
                        score = self.calculate_similarity(probe_emb, gallery_emb)
                        gallery_scores.append(score)
                        gallery_ids.append(gallery_id)

                #order with the descending score
                sorted_indices = np.argsort(gallery_scores)[::-1]
                sorted_ids = [gallery_ids[i] for i in sorted_indices]

                #find the rank of the correct match
                try:
                    correct_rank = sorted_ids.index(probe_id)+1 #first rank
                    ranks.append(correct_rank)
                except ValueError:
                    ranks.append(max_rank+1) # not found in the first ranks

        # calculate the CMS
        cms_scores = []
        for rank in range(1, max_rank + 1):
            accuracy = np.sum(np.array(ranks) <= rank) / len(ranks) #calculates how many probes have the correct match in the rank N, and to the %
            cms_scores.append(accuracy)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_rank + 1), cms_scores, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Rank')
        plt.ylabel('Recognition Rate')
        plt.title('Cumulative Match Score (CMS)')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.show()

        return cms_scores

    #plotting all the graphics
    def generate_all_plots(self):
        print("\n Generating all plots...")
        # 1. Score Distributions
        print(" Generating Score Distributions...")
        self.plot_score_distributions()
        # 2. ROC Curve
        print(" Generating ROC Curve...")
        auc_score = self.plot_roc_curve()
        # 3. DET Curve
        print(" Generating DET Curve...")
        self.plot_det_curve()
        # 4. FMR vs FNMR
        print(" Generating FMR vs FNMR...")
        eer, eer_threshold = self.plot_fmr_fnmr()
        # 5. CMS Curve
        print(" Generating CMS Curve...")
        cms_scores = self.plot_cms_curve()

        return {
            'auc': auc_score,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'rank1_accuracy': cms_scores[0],
            'rank5_accuracy': cms_scores[4]
        }

    #saving the results
    def save_results(self, filename='face_recognition_results.pkl'):
        results = { #takes all the important dates
            'similarity_scores': self.similarity_scores,
            'labels': self.labels,
            'detailed_results': self.detailed_results,
            'gallery_embeddings': self.gallery_embeddings,
            'probe_embeddings': self.probe_embeddings
        }
        with open(filename, 'wb') as f: #write binary
            pickle.dump(results, f) #saves in a file
        print(f" Results saved in {filename}")

    #loading the results
    def load_results(self, filename='face_recognition_results.pkl'):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        self.similarity_scores = results['similarity_scores']
        self.labels = results['labels']
        self.detailed_results = results['detailed_results']
        self.gallery_embeddings = results['gallery_embeddings']
        self.probe_embeddings = results['probe_embeddings']
        print(f" Results loaded from {filename}")