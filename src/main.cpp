#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ximgproc/lsc.hpp>

#include "object_recognition.h"

using namespace std;
using namespace cv;

/**
 * Splits the image in more images one for each cluster. We have K cluster centers and we'll return K mat istances.
 *
 * -Param Image is the original image
 * -Param Clustered is the clustered image (i.e. an image with k colors for each cluster)
 * -Param Centers contains cluster centers
 * -Param clusters is output vector which will contain splitted clusters
 */
void splitCluster(Mat image, Mat clustered, Mat1f centers, vector<Mat> &clusters) {
    for (int k = 0; k < centers.rows; k++) {
        Vec3b clusterColor;
        Vec6f ctr = centers.at<Vec6f>(k);
        clusterColor = Vec3b(ctr[2], ctr[3], ctr[4]);
        
        Mat mat = Mat(image.rows, image.cols, image.type(), Scalar(255, 255, 255));
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                Vec3b color = clustered.at<Vec3b>(i, j);
                if (color == clusterColor) {
                    mat.at<Vec3b>(i, j) = image.at<Vec3b>(i, j); //On each mat, we don't return the cluster color but the image's original color in corresponding clustered color.
                }
            }
        }
        clusters.push_back(mat);
    }
    
}

/**
 * Clusterize the image by position and color. The cluster
 *
 * -Param input is the image to clusterize
 * -Param output is the clustered image
 * -Param K is the number of centers
 * -Param centers is the output mat which contains the found centers of clusters
 *
 * -Returns the vector of mat which contains for each a cluster of the original image
 */
vector<Mat> cluster(Mat &input, Mat &output, int K, Mat1f &centers) {
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.01);
    vector<int> labels;
    Mat image = input.clone();
    //GaussianBlur(input, image, Size(7, 7), 2);
    
    float posMul = 500; //This ensures that the maximal value of positions if 500
    vector<Vec6f> to_cluster;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            //Normalized value of x and y multiplied by posMul in order to make the position relevance independet by the size of the image
            float x = static_cast<float>(j) / input.cols * posMul;
            float y = static_cast<float>(i) / input.rows * posMul;
            
            
            Vec3b color = image.at<Vec3b>(i, j);
            Vec6f point = Vec6f(x, y, color[0], color[1], color[2], 0); //We need a 5f vector. We use Vec6f but the last paramether is set to 0
            
            to_cluster.push_back(point);
        }
    }
    
    kmeans(to_cluster, K, labels, criteria, 1, KMEANS_PP_CENTERS, centers);
    
    output = image.clone();
    
    vector<Vec3b> colors;
    
    //Create color for each cluster
    for (int i = 0; i < K; i++) {
        
        float b = centers[i][2];
        float g = centers[i][3];
        float r = centers[i][4];
        
        Vec3b color = Vec3b(b, g, r);
        colors.push_back(color);
    }
    
    
    //Coloring the image with cluster colors in order to visualize the clustering
    size_t arrayIndex = 0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int center_id = labels[arrayIndex];
            Vec3f center = colors[center_id];
            output.at<Vec3b>(i, j) = center;
            arrayIndex++;
        }
    }
    
    
    //We return clustered images
    vector<Mat> clusters;
    splitCluster(input, output, centers, clusters);
    return clusters;
}

/**
 * Draws a rect that contains all pixels of the passed cluster in the original image
 *
 * -Param image is the original image
 * -Param cluster is the mat which contains one of K found clusters
 * -Param color is the color of the rect
 * -Param rect_width is the width of the desired rect
 */
void drawRect(Mat image, Mat cluster, Scalar color = Scalar(0, 255, 0), uint rect_width = 5) {
    float minX = FLT_MAX;
    float maxX = FLT_MIN;
    float minY = FLT_MAX;
    float maxY = FLT_MIN;
    for (int i = 0; i < cluster.rows; i++) {
        for (int j = 0; j < cluster.cols; j++) {
            Vec3b color = cluster.at<Vec3b>(i, j);
            if (color == Vec3b(255, 255, 255)) continue;
            
            if (j < minX) {
                minX = j;
            }
            if (j > maxX) {
                maxX = j;
            }
            if (i < minY) {
                minY = i;
            }
            if (i > maxY) {
                maxY = i;
            }
        }
    }
    
    rectangle(image, Point2f(minX, minY), Point2f(maxX, maxY), color, rect_width);
}

#define SHOULD_TRAIN false //Since the user that uses this program shouldn't be aware of what is going under the hood, the training process can be activated only in code. So when the user uses the program, a vocabulary and svm model must be already trained. To create models for user's program, the developer will activate the train phase directly in code.

//The input program should have as parameters image_path and number_of_clusters. If image_path is not passed, a default image will be used. When number_of_clusters is not passed, a default number of clusters is used (i.e. 5)
int main(int argc, char *argv[]) {
    
    ObjectRecognition treeFeature = ObjectRecognition(200, "vocabulary", "model");
    
#if SHOULD_TRAIN
    vector<String> images_paths;
    utils::fs::glob("../dataset", "*.*", images_paths);
    vector<Mat> images;
    for (String imPath : images_paths) {
        Mat mat = imread(imPath);
        
        if (mat.empty()) {
            continue;
        }
        images.push_back(mat);
    }
    
    
    int vocabularyImagesSize = 30;
    if (vocabularyImagesSize > images.size()) {
        cerr << "The number of images should be at least 30" << endl;
        return -1;
    }
    
    vector<Mat> vocabularyImages;
    for (int i = 0; i < vocabularyImagesSize; i++) {
        int randomIndex = theRNG().uniform(0, images.size());
        vocabularyImages.push_back(images[randomIndex]);
        
    }
    treeFeature.createVocabulary(vocabularyImages);
    treeFeature.train(images);
    cout << "Tree training ended" << endl;
#endif
    
    string imagePath;
    if (argc == 1) {
        imagePath = "../benchmark/Figure 5.jpg";
        
    } else {
        imagePath = string(argv[1]);
    }
    
    //Parsing the path of the image to predict
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "An image isn't passed" << endl;
        return -1;
    }
    
    //Parsing the number of clusters from input
    uint numberOfClusters = 5;
    if (argc == 3) {
        int inputCluster = atoi(argv[2]);
        if (inputCluster <= 0) {
            cerr << "Number of clusters must be > 0" << endl;
            return -1;
        }
        numberOfClusters = inputCluster;
    }
    
    
    
    Mat clusterImage;
    Mat1f clusterCenters;
    vector<Mat> clusters = cluster(image, clusterImage, numberOfClusters, clusterCenters);
    imshow("Clustered image", clusterImage);
    
    for (int i = 0; i < clusters.size(); i++) {
        Mat cluster = clusters[i];
        int res = treeFeature.predict(cluster);
        if (res == -1) { //If we are trying to predict with an untrained model, we should end the program
            return -1;
        }
        if (res == 1) {
            medianBlur(cluster, cluster, 11); //Median blur is used to remove pixel noise
            drawRect(image, cluster);
            
            //imshow("Cluster with tree " + to_string(i), cluster);
        }
    }
    
    
    imshow("Tree detection", image);
    
    waitKey(0);
    
    return 0;
}

