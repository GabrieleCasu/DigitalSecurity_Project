from FaceRecognitionProgram import FaceRecognition

if __name__ == "__main__":

    use_saved_results = True #if we want to use the already generated results (True)

    # initialize the system
    system = FaceRecognition(device='cpu')

    if use_saved_results:
        system.load_results("face_recognition_results.pkl")
    else:
     #paths
     gallery_path = "images/gallery"
     probe_path = "images/probe"
     # loading the datasets
     system.load_gallery_set(gallery_path)
     system.load_probe_set(probe_path)
     # execute the matching
     system.perform_matching(similarity_metric='cosine')  # it can be changed to 'euclidean' or 'manhattan'
     #save the results
     system.save_results("face_recognition_results.pkl")

     # Generate all the graphics
    results = system.generate_all_plots()

    print("\n" + "------------------------------------------")
    print(" Final results:")
    print(f"  AUC Score: {results['auc']:.3f}")
    print(f"️  Equal Error Rate: {results['eer']:.3f}")
    print(f" Threshold: {results['eer_threshold']:.3f}")
    print(f"  Rank-1 Accuracy: {results['rank1_accuracy']:.3f}")
    print(f"  Rank-5 Accuracy: {results['rank5_accuracy']:.3f}")

