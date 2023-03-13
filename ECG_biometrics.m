
%% Main script %%
%% LSTM networks for ECG-based biometric authentication systems %%
% Roberto Hernández Ruiz

%% 1. Load data: read .mat file and extract ECG signals

if ~isfile('ECGSignalsData.mat')
    % Read .csv file with subjects and load signals
    ReadECGData      
end
% .mat file with ECG signals + associated labels
load ECGSignalsData

% n_subjects = n_classes
num = size(Classes,1); 

for j=1:numSubjects
    
   % For each user userN = 1xn, create a var: n ECG signal samples
   eval(sprintf('user%d = Signals{j};',j));
   
   % Representation of ECG signals
%    figure(1)
%    subplot(round(numSubjects/2),2,j)
%    plot(Signals{j})
%    title(strcat('User',{' '},num2str(j)))
%    xlim([4000,5200])
%    xlabel('Samples')
%    ylabel('ECG (mV)')  
  
end

%% 2. Segmentation and windowing of ECG signals
% Divide ECG signal in windows of 9000 samples each to standardize
% Also makes number of instances to train network bigger
[Windows,Labels] = segmentSignals(Signals,Classes);

%% Histogram of ECG subjects signals length (opt)

L = cellfun(@length,Signals);
h = histogram(L);
xticks(0:500000:12000000);
xticklabels(0:500000:12000000);
title('Signal Lengths')
xlabel('Length')
ylabel('Number of users')

%% 3. After increasing number of instances per person to train the neural network,
% fixing window number or number of instances for all labels is needed

numWindows_Train = [];
numWindows_Test = [];

for j=1:numSubjects
    
    % Signal windows (userX) for each user with the associated label (userY)
    userX{j,1} = Windows(Labels==(strcat('User',num2str(j))));
    userY{j,1} = Labels(Labels==(strcat('User',num2str(j))));
    
    % Number of instances = windows generated per user
    windowsByUser(j) = size(userX{j,1},1);
    
    % Divide user instances = windows in trainset (80%) and testset (20%)
    [trainIndUser{j,1},~,testIndUser{j,1}] = dividerand(windowsByUser(j),0.8,0,0.2);
    
    % Number of windows for Train/Test after splitting
    nTrain = cell2mat(trainIndUser(j));
    nTest = cell2mat(testIndUser(j));
    
    % ECG signal (userX) + label (userY) of each subject
    x = userX(j);
    y = userY(j);
    
    % Split 0.8n windows (nTrain) for Train and 0.2n windows (ntest)
    % for Test, being n the number of windows per user
    XTrainUser{j,1} = x{1}(nTrain);
    YTrainUser{j,1} = y{1}(nTrain);
    XTestUser{j,1} = x{1}(nTest);
    YTestUser{j,1} = y{1}(nTest);
    
    % Concatenate number of windows for each user in Train, Test
    numWindows_Train = [numWindows_Train size(XTrainUser{j},1)];
    numWindows_Test = [numWindows_Test size(XTestUser{j},1)];
    
end

% Obtain minimum number of windows in Train and Test of all users (truncate to the minimum)
minTrain = min(numWindows_Train);
minTest = min(numWindows_Test);

%% 4. Data Preparation (XTrain, XTest) and labels (YTrain, YTest) for LSTM NN

XTrain = [];
YTrain = [];
XTest = [];
YTest = [];

for j=1:numSubjects
    
    % Obtain complete training set and test set,
    % using same number of windows for all users (minimum) 
    XTrain = [XTrain XTrainUser{j}(1:minTrain)];
    YTrain = [YTrain YTrainUser{j}(1:minTrain)];
    
    XTest = [XTest XTestUser{j}(1:minTest)];
    YTest = [YTest YTestUser{j}(1:minTest)];
    
end

% Flatten matrixes to column vectors to fit LSTM network 
XTrain = reshape(XTrain,[],1);
YTrain = reshape(YTrain,[],1);
XTest = reshape(XTest,[],1); 
YTest = reshape(YTest,[],1); 

%% 5.  Build LSTM network architecture

layers = [ ...
    sequenceInputLayer(1) % 1-D input signals
    bilstmLayer(100,'OutputMode','last') % map signals to 100 features
    fullyConnectedLayer(numSubjects) 
    softmaxLayer
    classificationLayer
    ]

% Técnica optimización ADAM
options = trainingOptions('adam', ...
    'MaxEpochs',10, ... 
    'MiniBatchSize', 150, ... % 150 signals at same time
    'InitialLearnRate', 0.01, ... 
    'SequenceLength', 1000, ... % signal segmentation
    'GradientThreshold', 1, ... % avoid too large gradients
    'ExecutionEnvironment',"auto",... 
    'plots','training-progress', ... % (accuracy vs iterations) training progress
    'Verbose',false); 

%% 6. Fit the networl

net = trainNetwork(XTrain,YTrain,layers,options);


%% 7. Classify Train signals, see accuracy and matrix confusion
% See how network works with already seen data

trainPred = classify(net,XTrain,'SequenceLength',1000);
LSTMAccuracy_Train = sum(trainPred == YTrain)/numel(YTrain)*100

figure
confusionchart(YTrain,trainPred,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Confusion Chart for LSTM - Train');

%% 8. Test set signals classification

testPred = classify(net,XTest,'SequenceLength',1000);
LSTMAccuracy_Test = sum(testPred == YTest)/numel(YTest)*100

figure
confusionchart(YTest,testPred,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Confusion Chart for LSTM - Test');

%% 9. APPROACH 2 (FEATURE EXTRACTION & TIME-FREQUENCY ANALYSIS)

% Sampling frequency (Hz)
fs = 300;

% TEMPORAL FREQUENCY MOMENTS: instant frequency and spectral entropy
for j=1:numSubjects
    
    [instFreqUser{j,1},tUser{j,1}] = instfreq(eval(sprintf('user%d',j)),fs);
    [pentropyUser{j,1},t2User{j,1}] = pentropy(eval(sprintf('user%d',j)),fs);
end

%% 10. Compute instfreq and pEntropy for each cell in dataset
% Save in different variables for concatenating

instfreqTrain = cellfun(@(x)instfreq(x,fs)',XTrain,'UniformOutput',false);
instfreqTest = cellfun(@(x)instfreq(x,fs)',XTest,'UniformOutput',false);

pentropyTrain = cellfun(@(x)pentropy(x,fs)',XTrain,'UniformOutput',false);
pentropyTest = cellfun(@(x)pentropy(x,fs)',XTest,'UniformOutput',false);

%% 11. Concatenate features of same sets

% Now, signal is 2-D: instFreq (row 1) y pEntropy (row 2) for each window
% (2 dim x 255 values of 255 temporal windows)
XTrain2 = cellfun(@(x,y)[x;y],instfreqTrain,pentropyTrain,'UniformOutput',false);
XTest2 = cellfun(@(x,y)[x;y],instfreqTest,pentropyTest,'UniformOutput',false);

% Mean of each feature for each subject
% They both differ in magnitude order! Be awware of this
for j=1:numSubjects
    meanInstFreqUser{j,1} = mean(instFreqUser{j,1});
    meanPEntropyUser{j,1} = mean(pentropyUser{j,1});
end

%% 12. Data standardization

XV = [XTrain2{:}];
mu = mean(XV,2);
sg = std(XV,[],2);

% New training set XTrainSD and test set XTestSD (already standardized)
XTrainSD = XTrain2;
XTrainSD = cellfun(@(x)(x-mu)./sg,XTrainSD,'UniformOutput',false);

XTestSD = XTest2;
XTestSD = cellfun(@(x)(x-mu)./sg,XTestSD,'UniformOutput',false);

%% 13. More info about t/freq features from signal

instFreqNSD = XTrainSD{1}(1,:); %first row is instFreq
pentropyNSD = XTrainSD{1}(2,:); %second row is pentropy

mean(instFreqNSD)
mean(pentropyNSD)


%% 14. Build a new architecture for this different LSTM network

layers2 = [ ...
    sequenceInputLayer(2) % now, 2-D signals as input (instFreq + pentropy)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(numSubjects)
    softmaxLayer
    classificationLayer
    ]

% New training configurations
options2 = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

% Because of the lower length of the input signals (now extracted features,
% before raw datapoints from raw ECG signal), a much shorter training process could be expected

%% 15. New LSTM network training

net2 = trainNetwork(XTrainSD,YTrain,layers2,options2);

%% 16. Train signals classification, accuracy and matrix confusion

trainPred2 = classify(net2,XTrainSD);
LSTMAccuracy_Train_2 = sum(trainPred2 == YTrain)/numel(YTrain)*100

figure
confusionchart(YTrain,trainPred2,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Confusion Chart for LSTM - Train 2');

%% 17. Test real accuracy of the LSTM network with test set

testPred2 = classify(net2,XTestSD);
LSTMAccuracy_Test_2 = sum(testPred2 == YTest)/numel(YTest)*100
figure
confusionchart(YTest,testPred2,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Confusion Chart for LSTM - Test 2');







