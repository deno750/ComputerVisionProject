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

class TreeFeatureExtractor {
private:
    uint vocabularySize;
    string vocabularypath;
    string svmModelpath;
    cv::Mat vocabulary;
    cv::Ptr<cv::ml::SVM> svm;
    
public:
    TreeFeatureExtractor(uint vocabularySize, string vocabularyFileName = "", string svmModelFileName = "") {
        this->vocabularySize = vocabularySize;
        this->vocabularypath = !vocabularyFileName.empty() ? "../models/vocabulary/" +vocabularyFileName+ ".yml" : "";
        this->svmModelpath = !svmModelFileName.empty() ? "../models/svm/" +svmModelFileName+ ".xml" : "";
        cv::FileStorage readStorage = cv::FileStorage(svmModelpath, cv::FileStorage::READ);
        if (!svmModelpath.empty() && readStorage.isOpened()) {
            svm = cv::Algorithm::load<cv::ml::SVM>(svmModelpath);
        } else {
            svm = cv::ml::SVM::create();
            svm->setType(cv::ml::SVM::ONE_CLASS);
            svm->setNu(0.5);
        }
        readStorage.release();
        
    }
    
    cv::Mat getvocabulary() {
        return vocabulary;
    }
    
    void createVocabulary(std::vector<cv::Mat> dataset) {
        
        if (!vocabularypath.empty()) {
            cv::FileStorage readStorage = cv::FileStorage(vocabularypath, cv::FileStorage::READ);
            readStorage["vocabulary"] >> vocabulary;
            readStorage.release();
            if (!vocabulary.empty()) {
                cout << "Vocabulary already trained" << endl;
                return;
            }
        }
        
        cout << "Started training" << endl;
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat untrainedDescriptions;
         
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
        for(cv::Mat image : dataset) {
            extractor->detect(image, keyPoints);
            cv::Mat descriptors;
            extractor->compute(image, keyPoints, descriptors);
             
            untrainedDescriptions.push_back(descriptors);
             
        }
         
        cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER, 100, 0.001);
        vector<int> labels;
        
        cv::BOWKMeansTrainer trainer = cv::BOWKMeansTrainer(vocabularySize);
        trainer.add(untrainedDescriptions);
        vocabulary = trainer.cluster();
        cout << "Vocabulary train finished" << endl;
        if (!vocabularypath.empty()) {
            cv::FileStorage writeStorage = cv::FileStorage(vocabularypath, cv::FileStorage::WRITE);
            
            writeStorage << "vocabulary" << vocabulary;
            writeStorage.release();
            cout << "Vocabulary saved to file" << endl;
        }
    }
    
    void train(vector<cv::Mat> dataset) {
        if (svm->isTrained()) {
            cout << "Svm model is loaded from file" << endl;
            return;
        }
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
        cv::BOWImgDescriptorExtractor bowExtraction = cv::BOWImgDescriptorExtractor(extractor, matcher);
        if (vocabulary.rows == 0) {
            cerr << "The vocabulary must be created first!" << endl;
            return;
        }
        bowExtraction.setVocabulary(vocabulary);
        cout << "Svm started to train" << endl;
        
        cv::Mat descriptors_to_train;
        cv::Mat labels;
        for (int i = 0; i < dataset.size(); i++) {
            cv::Mat image = dataset[i];
            
            cout << "i " << i << endl;
            vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            detector->detect(image, keypoints);
            bowExtraction.compute(image, keypoints, descriptors);
            descriptors_to_train.push_back(descriptors);
            cv::Mat label = cv::Mat::ones(descriptors.rows, 1, CV_32SC1);
            
            labels.push_back(label);
        }
        descriptors_to_train.convertTo(descriptors_to_train, CV_32FC1);
        
        bool trained = svm->trainAuto(descriptors_to_train, cv::ml::ROW_SAMPLE, labels);
        if (trained) {
            cout << "Svm successfully trained" << endl;
            if (!svmModelpath.empty()) {
                svm->save(svmModelpath);
                cout << "Svm model saved to file" << endl;
            }
        } else {
            cerr << "Svm didn't train successfully" << endl;
        }
        
    }
    
    int predict(cv::Mat image, vector<cv::KeyPoint> keypoints = vector<cv::KeyPoint>()) {
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
        
        cv::BOWImgDescriptorExtractor bowExtraction = cv::BOWImgDescriptorExtractor(extractor, matcher);
        bowExtraction.setVocabulary(vocabulary);
        
        cv::Mat descriptors;
        
        if (keypoints.size() == 0)
            detector->detect(image, keypoints);
        
        bowExtraction.compute(image, keypoints, descriptors);
        
        int res = svm->predict(descriptors);
        return res;
    }
};

#endif /* feature_extraction_h */
