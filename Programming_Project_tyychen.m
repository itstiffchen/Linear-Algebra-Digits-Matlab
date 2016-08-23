% Tiffany Chen
% Student ID: 998840686
% MAT 167: digit recognition project
% May 24, 2016

% step 1
% part a
% goal: load file into matlab session

% it is mat file format
% contains 4 arrays
% train_patterns, test_patterns, (256x4649)
% which contain 16x16 pixel intensities to lie in [-1,1] range
% train_labels, test_labels (10x4649)
% which contain true information about the digit images
load('USPS.mat')

% part b
% goal: display first 16 images in train_patterns

for k = 1:16
    subplot(4,4,k) % create subplot
    img = reshape(train_patterns(:,k),16,16)' % the first 16 cols are the images
    imagesc(img) % create the image, needs to be reshaped, then transposed
    % to see the digit, need to reshape column (vector of length 256) to 
    % small matrix of 16x16, transpose it, then use imagesc function
end

% step 2
% goal: a 256x10 matrix train_aves that is the average digits of
% train_patterns

% want to sort all the patterns of one "kind" like all 1s, 2s, 5s
% average those by "kind"
% store into one column
% thus, there will be 10 columns since there are 10 digits.

% initialize the train_aves matrix 
train_aves = zeros(256,10);

% loop through 10 digits
for k = 1:10
    subplot(2,5,k) % create plot
    train_aves(:,k) = mean(train_patterns(:, train_labels(k,:)==1),2); % compute means of code given, 
    % which pools all the images of one "kind"
    % by matching with 1 are thes same digits
    % mean function calculates mean by each column
    x = reshape(train_aves(:,k), 16,16)' % since imagesc needs reshape, do so
    imagesc(x)
end


% step 3
% goal: conduct simplest classification computation

% part a
% goal: test_classif matrix of 10x4649 with 
% euclidean distance between each image in 
% test_patterns and mean digit images in train_aves 

% initialize the test_classif matrix 
test_classif = zeros(10, 4649) 

% loop through 10 digits 
for k = 1:10
    test_classif(k,:) = sum((test_patterns-repmat(train_aves(:,k),[1 4649])).^2);
    % calculates Euclidean distance, which is the difference 
    % between test_patterns and train_aves, squared, then summed
end

% part b
% goal: compute classification results into vector
% test_classif_res of 1x4649

% initialize the test_classif_res vector
test_classif_res = zeros(1, 4649)

% loop through the 4649 
for j = 1:4649
    [tmp, ind] = min(test_classif(:,j)); % gets the position index
    % of the minimum of each jth column of test_classif 
    test_classif_res(:,j) = ind; % the numbers corresponding to ind are the 
    % actual position indexes we care about
    % so to grab those, simply create a new variable that will equal 
    % all the ind values
end

% part c
% goal: compute confusion matrix test_confusion 10x10 

% initialize the test_confusion matrix
test_confusion = zeros(10, 10)

% confusion matrix
% loop through the labels rows
% and see if it matches 1 because that 
% means it is the digit
for k = 1:10
    tmp = test_classif_res(test_labels(k,:) == 1);
    % test_classif_res stores the prediction
    % test_labels stores the actual label
    % want to compare the two
    % if they match, that is good and will be on the diagonal
    % then loop through the columns 
    % to find what matches with the digit (the actual and predicted)
    for j = 1:10
        test_confusion(k,j) = sum(tmp(1,:) == j);
        % sum will matches it 
        % rows are test_labels (actual)
        % columns are predicted
    end
end
% it seems like majority of the numbers lie on the diagonal which is good
% that means most digits are accurately labeled, tho thtere are still a few
% inaccurate labels in the off diagonals

diag(1./sum(test_confusion, 2))*test_confusion
diag(1./sum(test_svd17_confusion, 2))*test_svd17_confusion
% step 4
% goal: conduct a SVD based classification computation

% part a
% goal: pool all images corres to kth digit in train_patterns
% compute rank 17 SVD of that set of images
% in train_u 256x17x10 
for k = 1:10
    [train_u(:,:,k),tmp,tmp2] = svds(train_patterns(:, train_labels(k,:)==1),17);
    % computes the svd of the images
    % specifically the svd 17 rank (code was given) 
end

%part b
% goal: compute expansion coeff wrt 17 singular vectors of each train 
% digit image set
% compute 17x10 numbers for each test digit image 
% into test_svd17 17x4649x10 
for k=1:10
    test_svd17(:,:,k) = train_u(:,:,k)'*test_patterns ;
    % expansion coeff is obtained by dot product of train_u and
    % test_patterns
end

% part c
% goal: compute error btwn original test digit image
% and its rank 17 approximation using kth digit images 
% in training data set
test_svd17res = zeros(10,4649) 

% we will use euclidean distance for the error
% another better way is to use least squares
% but that is a bit more difficult, so Puckett
% says we can use Euclidean distance 
% so, very similar to the previous steps
for k = 1:10
    test_svd17res(k,:) = sum((test_patterns-(train_u(:,:,k)*test_svd17(:,:,k))).^2);
    % subtract the code given from the test patterns, square, then sum
    % code given is the rank 17 approximations of test digits
    % store into test_svd17res 
end

% part d
% goal: compute confusion matrix using
% SVD based classification method 

% this loop first computes
for j = 1:4649
    [tmp, ind] = min(test_svd17res(:,j)); % gets the position index
    % of the minimum of each jth column of test_classif 
    test_classif_svd(:,j) = ind; % the numbers corresponding to ind are the 
    % actual position indexes we care about
    % so to grab those, simply create a new variable that will equal 
    % all the ind values
end

% this loops computes the 
% svd confusion matrix
% test_classif_svd stores predicted labels
% test_labels stores the actual labels
% need two for loops
for k = 1:10
    % this first loop 
    % goes through the labels and 
    % assigns what the actual label value is
    tmp2 = test_classif_svd(test_labels(k,:) == 1);
    for j = 1:10
        % this second loop
        % matches the predicted with actual
        % and keeps a count 
        test_svd17_confusion(k,j) = sum(tmp2(1,:) == j);
    end
end
% it is clear that the confusion matrix for svd is better than from step 3
% using the euclidean distance because the values in the diagonals are
% greater than in the euclidean

% step 5
% goal: analyze results!
% 
% part a
% Given a set of manually classified digits (training set), we want to 
% classify a set of unknown digits (test set). So in this dataset USPS, 
% there are a certain number of digits 9398, relatively equally distributed
% between 0 to 9 and the test set also has half 4649 and the training has 
% the other half 4649. 
% . A little background
% on the USPS dataset is that was gathered at the Center of Excellence in
% Doc Analysis and Recogntion at SUNY buffalo, sponsered by the USPS. 
% Train_labels and Test_labels is a 10x4649 matrix where the columns represent
% a certain digit from 0-9.Where the 1 is located in the column, 
% corresponds to what the digit actually is. More specifically, the arrays
% contain the true information about the digit images. Additionally, the
% jth handwritten digit image in train_patterns truly represents the digit
% i and the (i+1,j)th entry of train_labels is 1, while the other entries
% of the jth column is -1.
% The rest of the entries in the column is -1. The train_patterns and
% test_patterns are matrices of 256x4649 that contains numbers that
% represent a raster scan of the 16x16 gray level pixel intensities that
% have been normalized to lie withthe range [-1,1]. 
% 
% part b
% In step 2, we want to classify the unknown digits by using Euclidean
% distance. This means, in the training set, given the manually classified
% set of digits, we compute the means of all the 10 classes/kinds of
% digits. Then, for each digit in the test set, classify it as that digit
% if it is the closest mean (or closest in Euclidean distance). If we
% consider the training set digits as vectors or points, then we can assume
% the digits of one kind form clusters in a Euclidean space. These clusters
% are well separated if the training set digits are well written. From the
% output, the digits are mostly well written because we illustrate the
% means of the digits so the clusters are well separated. So perhaps using
% Euclidean distance classification algorithm is ok, but we will compare to
% SVD later on.
% 
% part c
% Overall the SVD algorithm is better than using the squared Euclidean
% distances because the values in the diagonals are greater than in the
% Euclidean case. For SVD, what we do is consider the images as 16x16
% matrices. Then the columns of a matrix consisting of all the training
% digits of one kind will span a linear subspace of R m. To model the
% variation within the training and test sets, we use orthogonal basis of
% the subspace. The SVD assumptions are that each digit is well
% characterized by a few of the first singular images of its own kind and
% that the expansion coefficients discriminates well between the different
% classes of digits. Also, if the unknown digit is well classified for one
% particular basis than another, then its likely the unknown digit is that
% number. For the training set of known digits, compute the SVD of each set
% of digits. 
% 
% Looking at the percentage of each entry accuracy, it is obvious that SVD
% algorithm is better than Euclidean. Eculidean overall classification rate
% is 84.66% 
% Euclidean:  see matrix
diag(1./sum(test_confusion, 2))*test_confusion
% SVD: see matrix below
diag(1./sum(test_svd17_confusion, 2))*test_svd17_confusion
% the digit most difficult to identify correctly is 5 for euclidean and 8
% for SVD. 
% the digit easiest to identify correctly is 2 for SVD and euclidean.
% 
% part d
% The SVD classification yields better results than Euclidean distance. For
% more detailed explanation, see the pdf file i'll write up!
