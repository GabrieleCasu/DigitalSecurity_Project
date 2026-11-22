Big age gap:
Identity 3 (no problem)
Identity 10 (no problem)
Identity 13 (too low results)
Identity 28 (no problem)


Impostor:
Identity 24
Identity 14
Identity 12
Identity 6
Identity 8


Parameter used to decide the gallery image:
High quality
Front faced
Good illumination
Neutral pose

IdentityN02_18 was eliminated because it causes problems during the embedding phase

All the tests with different parameters value have been done with the cosine metric

mtcnn
keep_all=False //it uses only one face per image, based of the method, if we put true it takes all the faces
post_process // default=true so i don't put it, if i put post_process=False the results are significantly worse, like 0.600 accuracy
1)min_face_size //default=20 the minimum size a face has to be to be recognized, with 1.1)40 almost the same results but two embedding errors, with 1.2)10 much slower and two error with the embedding and a little bit less precise
4)thresholds=[t1,t2,t3] //default[0.6,0.7,0.7] and zero embedding errors , this are general recognition, more accurate and even more accurate, same results but decreasing it more images gives embedding errors and potentially more false positives, increasing it we have more false negatives
3)factor //default=0.709, with every scale we reduce the image of the 70.9%, 3.2)with 0.5 is slower but has more scales, with 3.1)0.9 is less accurate but faster with very big/small faces but with 1 embedding error
2)selection_method// to select the face if keep_all=false, default=probability, with largest we take the biggest which does the same results

InceptionResnetV1
pretrained='vggface2' //big dataset, there are others like 'casia-webface' but the results are the same
classify=True //it's useful if we want to train the ai on our dataset
num_classes //number of persons that we use to train the ai
dropout_prob //random deactivation of some neurons, used only if we are training the ai