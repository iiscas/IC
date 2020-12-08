close all; clear; clc
featureDim=10;

% TREINO
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

% figure; colormap(gray)
% for i=1:25
%     subplot(5,5,i)
%     digit = reshape(images(:,i),[28 28]);
%     imagesc(digit)
% end

conjuntoDeTreino_x = images;                               
conjuntoDeTreino_y_temp = labels;                           


for sample = 1 : length(conjuntoDeTreino_y_temp')            % sample é cada amostra
      conjuntoDeTreino_y(:,sample) = (conjuntoDeTreino_y_temp(sample)==0:9);
 end

% fprintf('Conjunto de treino X: %d %d\n',size(conjuntoDeTreino_x));
% fprintf('Conjunto de treino Y: %d %d\n',size(conjuntoDeTreino_y));

% TESTE
images = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

% figure; colormap(gray)
% for i=1:25
%     subplot(5,5,i)
%     digit = reshape(images(:,i),[28 28]);
%     imagesc(digit)
% end

conjuntoDeTeste_x = images;    
conjuntoDeTeste_y_temp = labels';    


for sample = 1 : length(conjuntoDeTeste_y_temp)            % sample é cada amostra
    conjuntoDeTeste_y(:,sample) = (conjuntoDeTeste_y_temp(sample)==0:9);
end

% fprintf('Conjunto de teste X: %d %d\n',size(conjuntoDeTeste_x));
% fprintf('Conjunto de teste Y: %d %d\n',size(conjuntoDeTeste_y));


net= treinoMLP(conjuntoDeTreino_x,conjuntoDeTreino_y,conjuntoDeTeste_x,conjuntoDeTeste_y);