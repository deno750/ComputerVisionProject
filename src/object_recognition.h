//
//  feature_extraction.h
//  Project
//
//  Created by Denis Deronjic on 10/06/2020.
//
#ifndef feature_extraction_h
#define feature_extraction_h

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/ml.hpp>

using namespace std;

class ObjectRecognition {
private:
    string vocabularypath;
    string svmModelpath;
    cv::Mat vocabulary;
    cv::Ptr<cv::ml::SVM> svm;
    cv::Ptr<cv::FeatureDetector> detector;
    
    //Bag of words extractor which is initialized with SiftDescriptorExtractor and FlanBasedMatcher
    cv::BOWImgDescriptorExtractor bowExtraction = cv::BOWImgDescriptorExtractor(cv::xfeatures2d::SiftDescriptorExtractor::create(),
                                                                                cv::FlannBasedMatcher::create());
    
public:
    /**
     * -Param vocabularyFileName is the file name of the vocabulary if we want to store it locally
     * -Param svmModelFileName is the file name of the SVM model if we want to store it locally
     */
    ObjectRecognition(string vocabularyFileName = "", string svmModelFileName = "") {
        this->vocabularypath = !vocabularyFileName.empty() ? "../models/vocabulary/" +vocabularyFileName+ ".yml" : "";
        this->svmModelpath = !svmModelFileName.empty() ? "../models/svm/" +svmModelFileName+ ".xml" : "";
        detector = cv::xfeatures2d::SiftFeatureDetector::create();
        
        cv::FileStorage svmReadStorage = cv::FileStorage(svmModelpath, cv::FileStorage::READ);
        if (!svmModelpath.empty() && svmReadStorage.isOpened()) { //We check if a SVM model is already trained with the name passed in svmModelFileName
            svm = cv::Algorithm::load<cv::ml::SVM>(svmModelpath);
        } else {
            //A one class SVM is created
            svm = cv::ml::SVM::create();
            svm->setType(cv::ml::SVM::ONE_CLASS);
            svm->setNu(0.5);
        }
        svmReadStorage.release();
        
        cv::FileStorage vocabularyReadStorage = cv::FileStorage(vocabularypath, cv::FileStorage::READ);
        if (!vocabularypath.empty() && vocabularyReadStorage.isOpened()) { //Checks wheter a vocabulary is already stored
            vocabularyReadStorage["vocabulary"] >> vocabulary;
            bowExtraction.setVocabulary(vocabulary);
        }
        vocabularyReadStorage.release();
        
    }
    
    cv::Mat getvocabulary() {
        return vocabulary;
    }
    
    /**
     * Creates the BoW vocabulary using the passed dataset
     * -Param dataset is the dataset for training the vocabulary
     * -Param vocabularySize defines the size of the BoW vocabulary
     */
    void createVocabulary(std::vector<cv::Mat> dataset, uint vocabularySize) {
        if (!vocabulary.empty()) {
            cout << "Vocabulary already trained" << endl;
            return;
        }
        
        cout << "Started vocabulary training" << endl;
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat untrainedDescriptions;
        
        //Extracts descriptors of images in the dataset
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
        for(cv::Mat image : dataset) {
            extractor->detect(image, keyPoints);
            cv::Mat descriptors;
            extractor->compute(image, keyPoints, descriptors);
             
            untrainedDescriptions.push_back(descriptors);
             
        }
         
        cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER, 100, 0.001);
        
        //Trains the vocabulary using kmeans clustering
        cv::BOWKMeansTrainer trainer = cv::BOWKMeansTrainer(vocabularySize);
        trainer.add(untrainedDescriptions);
        vocabulary = trainer.cluster();
        bowExtraction.setVocabulary(vocabulary);
        cout << "Vocabulary train finished" << endl;
        
        //Stores the vocabulary on a file
        if (!vocabularypath.empty()) {
            cv::FileStorage writeStorage = cv::FileStorage(vocabularypath, cv::FileStorage::WRITE);
            
            writeStorage << "vocabulary" << vocabulary;
            writeStorage.release();
            cout << "Vocabulary saved to file" << endl;
        }
    }
    
    /**
     * Trains an SVM model using the passed dataset
     */
    void train(vector<cv::Mat> dataset) {
        if (svm->isTrained()) {
            cout << "Svm model is loaded from file" << endl;
            return;
        }
        
        if (vocabulary.rows == 0) {
            cerr << "The vocabulary must be created first!" << endl;
            return;
        }
       
        cout << "Svm started to train" << endl;
        
        cv::Mat descriptors_to_train;
        vector<int> labels;
        //We extract the descriptors for each image in the dataset using BoW extraction rather than using SIFT or any other feature extraction.
        for (int i = 0; i < dataset.size(); i++) {
            cv::Mat image = dataset[i];
            
            cout << "i " << i << endl;
            vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            detector->detect(image, keypoints); //We compute keypoints that are used for training descriptors
            bowExtraction.compute(image, keypoints, descriptors); //Extract the descriptors of an image using BoW
            descriptors_to_train.push_back(descriptors);
             
            if (!descriptors.empty())
                labels.push_back(1); //Ones label which tells that this image is a positive image that contains the object we want recognize
        }
        descriptors_to_train.convertTo(descriptors_to_train, CV_32FC1);
        
        bool trained = svm->trainAuto(descriptors_to_train, cv::ml::ROW_SAMPLE, labels); //Trains SVM with the computed descriptors
        if (trained) {
            cout << "Svm successfully trained" << endl;
            if (!svmModelpath.empty()) { //Stores the SVM model to file
                svm->save(svmModelpath);
                cout << "Svm model saved to file" << endl;
            }
        } else {
            cerr << "Svm didn't train successfully" << endl;
        }
        
    }
    
    /**
     * Predicts wheter the passed image contains the desired object.
     *
     * -Param image the image where we want detect the desired object
     * -Param keypoints is used when keypoints of an image are already available, so we can pass them here to avoid a recomputation. If this parameter is not passed, the keypoints will be computed inside this function
     *
     * -Returns 1 if the image contains the expected object, 0 otherwise. -1 if the prediction is impossible
     */
    int predict(cv::Mat image, vector<cv::KeyPoint> keypoints = vector<cv::KeyPoint>()) {
        if (vocabulary.empty() || !svm->isTrained()) {
            cerr << "You can't predict an image without a trained vocabulary and a prediction model" << endl;
            return -1;
        }
        cv::Mat descriptors;
        
        if (keypoints.size() == 0)
            detector->detect(image, keypoints);
        
        bowExtraction.compute(image, keypoints, descriptors);
        
        int res = svm->predict(descriptors);
        return res;
    }
};

#endif /* feature_extraction_h */
