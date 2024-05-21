clc
close all
warning off
[filename, pathname] = uigetfile('*.*', 'Pick a MATLAB code file');
filename=strcat(pathname,filename);
a=imread(filename);
imshow(a);
b=rgb2gray(a);
figure;
imshow(b);
title("RGB image");
impixelinfo;
c=imbinarize(b,20/255);
figure;
imshow(c);
title("Binarizes the grayscale image ");
d=imfill(c,'holes');
figure;
imshow(d);
title("Fills the holes in the binarized image");
e=bwareaopen(d,1000);
figure;
imshow(e);
title("Removing small objects");
PreprocessedImage=uint8(double(a).*repmat(e,[1,1,3]));
figure;
imshow(PreprocessedImage);
title("Preprocessed Image");
PreprocessedImage=imadjust(PreprocessedImage,[0.3 0.7],[])+50;
figure;
imshow(PreprocessedImage);
title("Adjust Preprocessed Image");
uo=rgb2gray(PreprocessedImage);
figure;
imshow(uo);
title("RGB of Preprocessed Image");
mo=medfilt2(uo,[5 5]);
figure;
imshow(mo);
title("Median Filter");
impixelinfo;
po=mo>250;
figure;
imshow(po);
title("Binary Mask");
[r, c, m ]=size(po);
x1=r/2;
y1=c/3;
row=[x1 x1+200 x1+200 x1];
col=[y1 y1 y1+40 y1+40];
BW=roipoly(po,row,col);
figure;
imshow(BW);
title("ROI");
k=po.*double(BW);
figure;
imshow(k);
title("Isloated ROI");
M=bwareaopen(k,4);
[ya, number]=bwlabel(M);
if(number>=1)
    disp('Stone is detected:');
else
    disp('No stone is detected:');
end
% Step 1: Data Preparation
image_folder = uigetfile('*.*', 'Pick a MATLAB code file');
annotation_folder = 'path_to_annotations';
image_files = dir(fullfile(image_folder, '*.jpg'));
num_images = length(image_files);

annotations = zeros(num_images, 1);
for i = 1:num_images
    annotation_file = fullfile(annotation_folder, [image_files(i).name '.txt']);
    annotations(i) = load(annotation_file);
end

% Step 2: Feature Extraction (DWT)
wavelet = 'db4';
level = 3;

all_dwt_features = [];
for i = 1:num_images
    image_path = fullfile(image_folder, image_files(i).name);
    image = imread(image_path);
    
    [C, S] = wavedec2(image, level, wavelet);
    
    dwt_features = [];
    for j = 1:level
        dwt_coeff = appcoef2(C, S, wavelet, j);
        dwt_features = [dwt_features, dwt_coeff(:)'];
    end
    
    all_dwt_features = [all_dwt_features; dwt_features];
end

% Step 3: Simple Classification (Support Vector Machine - SVM)
% Assume you have a portion of the data for training and testing
% You'll need to split the data and annotations accordingly

% Generate synthetic data for demonstration purposes
num_samples = 100;
num_features = 27; % Adjust this based on your DWT feature dimension

% Generate random DWT features (replace this with your actual features)
train_features = rand(num_samples, num_features);
test_features = rand(num_samples, num_features);

% Generate random binary labels (0 or 1)
train_labels = randi([0, 1], num_samples, 1);
test_labels = randi([0, 1], num_samples, 1);

% Convert numeric labels to categorical
train_labels = categorical(train_labels);
test_labels = categorical(test_labels);

% Specify class names
class_names = categories(train_labels);

% Train SVM classifier
svm_model = fitcsvm(train_features, train_labels, 'ClassNames', class_names);

% Predict using SVM
predicted_labels = predict(svm_model, test_features);

% Evaluate performance
accuracy = sum(predicted_labels == test_labels) / length(test_labels);
disp(['Accuracy: ' num2str(accuracy)]);
