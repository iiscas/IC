function [net] = treinoMLP(conjuntoDeTreino_x, conjuntoDeTreino_y, conjuntoDeTeste_x, conjuntoDeTeste_y)
% Treinar uma rede MLP e apresentar a matriz de confusão
numero_de_neuronios = 10;
coeficiente_aprendizagem =0.01;
drawOn =1 ;

% Inicializar a rede neuronal
net = patternnet(numero_de_neuronios);
%net=newff(conjuntoDeTreino_x,conjuntoDeTreino_y,numero_de_neuronios);
% Personalizar o algoritmo de treino
net.trainFcn = 'traingd'; %Gradient Descent
%net.trainFcn = 'trainlm'; %Levenberg-Marquardt --comentar lr
%net.trainParam.lr = coeficiente_aprendizagem; %learning rate
net.trainParam.epochs=1000;

% Treinar a rede
net = train(net, conjuntoDeTreino_x, conjuntoDeTreino_y);

% Calcular as saidas dadas pela rede no conjunto de teste 
saidaDaRedeParaConjuntoDeTeste = net(conjuntoDeTeste_x);


%simulação e avaliação da nn
Toutputs=round(sim(net,conjuntoDeTeste_x));
Erro=conjuntoDeTeste_y-Toutputs;

%number of hits (accurancy), use this to get the NN performance
acuracy=1-length(find(Erro))/length(Erro);

  
% Matriz de confusão
if drawOn == 1
    figure;
    plotconfusion(conjuntoDeTeste_y,saidaDaRedeParaConjuntoDeTeste);
    figure;
    plotroc(conjuntoDeTeste_y,saidaDaRedeParaConjuntoDeTeste); %ROC-->AUC
end 
     
%apanhar a matriz de confusao
[c,cm,ind,per] = confusion(conjuntoDeTeste_y,saidaDaRedeParaConjuntoDeTeste);


%------------------------------ AVALIAÇÃO DO DESEMPENHO DA REDE NEURONAL
totalteste=10000;
cmT = cm';
int = 0;

FN = zeros(10,1);
FP = zeros(10,1);
TP = zeros(10,1);
TN = zeros(10,1);


%Para cada classe tirar os TPs, TNs, FPs e FNs

% - Classe 0
TP(1,1) = cmT(1,1);

for int=1:10
    FN(1,1) = FN(1,1) + cmT(int,1);
    FP(1,1) = FP(1,1) + cmT(1,int);
end
FN(1,1)= FN(1,1)-TP(1,1);
FP(1,1) = FP(1,1)-TP(1,1);
TN(1,1)=totalteste-(FN(1,1)+FP(1,1)+TP(1,1));
%-------------------------------------------
% - Classe 1
TP(2,1) = cmT(2,2);
for int=1:10
    FN(2,1) = FN(2,1) + cmT(int,2);
    FP(2,1) = FP(2,1) + cmT(2,int);
end
 FN(2,1) = FN(2,1) -TP(2,1);
 FP(2,1) = FP(2,1) -TP(2,1);
 TN(2,1)=totalteste-(FN(2,1)+FP(2,1)+TP(2,1));
 %-------------------------------------------
% - Classe 2
TP(3,1) = cmT(3,3);
for int=1:10
    FN(3,1) = FN(3,1) + cmT(int,3);
    FP(3,1) = FP(3,1) + cmT(3,int);
end
 FN(3,1) = FN(3,1) -TP(3,1);
 FP(3,1) = FP(3,1) -TP(3,1);
 TN(3,1)=totalteste-(FN(3,1)+FP(3,1)+TP(3,1));
  %-------------------------------------------
% - Classe 3
TP(4,1) = cmT(4,4);
for int=1:10
    FN(4,1) = FN(4,1) + cmT(int,4);
    FP(4,1) = FP(4,1) + cmT(4,int);
end
 FN(4,1) = FN(4,1) -TP(4,1);
 FP(4,1) = FP(4,1) -TP(4,1);
 TN(4,1)=totalteste-(FN(4,1)+FP(4,1)+TP(4,1));
 %-------------------------------------------
 % - Classe 4
TP(5,1) = cmT(5,5);
for int=1:10
    FN(5,1) = FN(5,1) + cmT(int,5);
    FP(5,1) = FP(5,1) + cmT(5,int);
end
 FN(5,1) = FN(5,1) -TP(5,1);
 FP(5,1) = FP(5,1) -TP(5,1);
 TN(5,1)=totalteste-(FN(5,1)+FP(5,1)+TP(5,1));
  %-------------------------------------------
  % - Classe 5
TP(6,1) = cmT(6,6);
for int=1:10
    FN(6,1) = FN(6,1) + cmT(int,6);
    FP(6,1) = FP(6,1) + cmT(6,int);
end
 FN(6,1) = FN(6,1) -TP(6,1);
 FP(6,1) = FP(6,1) -TP(6,1);
 TN(6,1)=totalteste-(FN(6,1)+FP(6,1)+TP(6,1));
 %-------------------------------------------
   % - Classe 6
TP(7,1) = cmT(7,7);
for int=1:10
    FN(7,1) = FN(7,1) + cmT(int,7);
    FP(7,1) = FP(7,1) + cmT(7,int);
end
 FN(7,1) = FN(7,1) -TP(7,1);
 FP(7,1) = FP(7,1) -TP(7,1);
 TN(7,1)=totalteste-(FN(7,1)+FP(7,1)+TP(7,1));
 %-------------------------------------------
   % - Classe 7
TP(8,1) = cmT(8,8);
for int=1:10
    FN(8,1) = FN(8,1) + cmT(int,8);
    FP(8,1) = FP(8,1) + cmT(8,int);
end
 FN(8,1) = FN(8,1) -TP(8,1);
 FP(8,1) = FP(8,1) -TP(8,1);
 TN(8,1)=totalteste-(FN(8,1)+FP(8,1)+TP(8,1)); 
  %-------------------------------------------
   % - Classe 8
TP(9,1) = cmT(9,9);
for int=1:10
    FN(9,1) = FN(9,1) + cmT(int,9);
    FP(9,1) = FP(9,1) + cmT(9,int);
end
 FN(9,1) = FN(9,1) -TP(9,1);
 FP(9,1) = FP(9,1) -TP(9,1);
 TN(9,1)=totalteste-(FN(9,1)+FP(9,1)+TP(9,1)); 
   %-------------------------------------------
   % - Classe 10
TP(10,1) = cmT(10,10);
for int=1:10
    FN(10,1) = FN(10,1) + cmT(int,10);
    FP(10,1) = FP(10,1) + cmT(10,int);
end
 FN(10,1) = FN(10,1) - TP(10,1);
 FP(10,1) = FP(10,1) - TP(10,1);
 TN(10,1)=totalteste-(FN(10,1)+FP(10,1)+TP(10,1));
 
 
 %----CALCULO DO ACURACY, PRECISAO,SENSIBILIDADE ESPECIFIDADE PARA CADA CLASSE 
for i=1:10
     
     
     tp=TP(i,1);
     fp=FP(i,1);
     fn=FN(i,1);
     tn=TN(i,1);
     
     acuracy=(TP(i,1)+TN(i,1))/totalteste;
     precisao=TP(i,1)/(TP(i,1)+FP(i,1));
     sensibilidade=TP(i,1)/(TP(i,1)+FN(i,1));
     especifidade=TN(i,1)/(FP(i,1)+TN(i,1));
     
        fprintf('Classe %d: ',i-1);
        fprintf('   TP = %d',tp);
        fprintf('   FP = %d',fp);
        fprintf('   FN = %d',fn);
        fprintf('   TN = %d',tn);

        fprintf('\nAcuracy = %0.2f\n',acuracy);
        fprintf('Precisao = %0.2f\n',precisao);
        fprintf('Sensibilidade = %0.2f\n',sensibilidade);
        fprintf('Especifidade = %0.2f\n',especifidade);
end