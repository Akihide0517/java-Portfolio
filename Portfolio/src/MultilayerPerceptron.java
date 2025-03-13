import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class MultilayerPerceptron extends MultilayerPerceptronAbstract{

    private int numFeatures; // 入力特徴量数
    private int numHiddenLayers; // 隠れ層数
    private int[] hiddenLayerSizes; // 各隠れ層のニューロン数。{4, 3} のように指定すれば、1層目の隠れ層は4ニューロン、2層目の隠れ層は3ニューロンとなる
    private int numOutputNodes; // 出力層のニューロン数
    private ActivationFunction activationFunction; // 活性化関数 (隠れ層用)
    private OutputActivationFunction outputActivationFunction; // 出力層の活性化関数 (Softmax用)
    private double learningRate;
    private double beta1; // Adamパラメータ
    private double beta2; // Adamパラメータ
    private double epsilon; // Adamパラメータ
    private double l2Regularization; // L2正則化係数

    private List<double[][]> weights; // 重み行列
    private List<double[]> biases; // バイアスベクトル
    private List<double[]> m_weights; // Adam用 モーメント (重み)
    private List<double[]> v_weights; // Adam用 モーメント (重み)
    private List<double[]> m_biases;  // Adam用 モーメント (バイアス)
    private List<double[]> v_biases;  // Adam用 モーメント (バイアス)
    private int t; // Adam用 タイムステップ
    
    // コンストラクタ (分類問題用、出力層にSoftmaxを使用)
    public MultilayerPerceptron(int numFeatures, int numHiddenLayers, int[] hiddenLayerSizes, int numOutputNodes, ActivationFunction activationFunction, OutputActivationFunction outputActivationFunction, double learningRate, double beta1, double beta2, double epsilon, double l2Regularization) {
        this.numFeatures = numFeatures;
        this.numHiddenLayers = numHiddenLayers;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.numOutputNodes = numOutputNodes;
        this.activationFunction = activationFunction;
        this.outputActivationFunction = outputActivationFunction; // Softmaxを設定
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.l2Regularization = l2Regularization;

        initializeWeightsAndBiases();
        initializeAdamParameters();
        this.t = 0;
    }

    // コンストラクタをnullにする
    public MultilayerPerceptron(int numFeatures, int numHiddenLayers, int[] hiddenLayerSizes, int numOutputNodes, ActivationFunction activationFunction, double learningRate, double beta1, double beta2, double epsilon, double l2Regularization) {
        this(numFeatures, numHiddenLayers, hiddenLayerSizes, numOutputNodes, activationFunction, null, learningRate, beta1, beta2, epsilon, l2Regularization);
    }

    // 重みとバイアスの初期化 (He初期値)
    private void initializeWeightsAndBiases() {
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        Random random = new Random();

        // 入力層 -> 最初の隠れ層
        weights.add(createWeightMatrix(numFeatures, hiddenLayerSizes[0], random));
        biases.add(new double[hiddenLayerSizes[0]]);

        // 隠れ層 -> 隠れ層
        for (int i = 0; i < numHiddenLayers - 1; i++) {
            weights.add(createWeightMatrix(hiddenLayerSizes[i], hiddenLayerSizes[i + 1], random));
            biases.add(new double[hiddenLayerSizes[i + 1]]);
        }

        // 最後の隠れ層 -> 出力層
        weights.add(createWeightMatrix(hiddenLayerSizes[numHiddenLayers - 1], numOutputNodes, random));
        biases.add(new double[numOutputNodes]);
    }

    // Heの初期値に基づいて重み行列を生成
    private double[][] createWeightMatrix(int numRows, int numCols, Random random) {
        double[][] matrix = new double[numRows][numCols];
        double stddev = Math.sqrt(2.0 / numRows);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                matrix[i][j] = random.nextGaussian() * stddev;
            }
        }
        return matrix;
    }

    // Adamパラメータの初期化
    private void initializeAdamParameters() {
        m_weights = new ArrayList<>();
        v_weights = new ArrayList<>();
        m_biases = new ArrayList<>();
        v_biases = new ArrayList<>();

        for(double[][] weightMatrix : weights) {
            m_weights.add(new double[weightMatrix.length * weightMatrix[0].length]);
            v_weights.add(new double[weightMatrix.length * weightMatrix[0].length]);
        }
        for(double[] biasVector : biases) {
            m_biases.add(new double[biasVector.length]);
            v_biases.add(new double[biasVector.length]);
        }
    }


    // 順伝播
    public double[] forwardPropagation(double[] inputs) {
        List<double[]> layerOutputs = new ArrayList<>();
        layerOutputs.add(inputs); // 入力層の出力

        double[] currentInputs = inputs;
        for (int layerIndex = 0; layerIndex < weights.size(); layerIndex++) {
            double[][] weightMatrix = weights.get(layerIndex);
            double[] biasVector = biases.get(layerIndex);
            // 線形結合の計算
            double[] weightedInput = new double[weightMatrix[0].length];
            for (int j = 0; j < weightMatrix[0].length; j++) {
                double sum = 0;
                for (int i = 0; i < weightMatrix.length; i++) {
                    sum += currentInputs[i] * weightMatrix[i][j];
                }
                sum += biasVector[j];
                weightedInput[j] = sum;
            }

            // 活性化関数の適用
            double[] layerOutput;
            if (layerIndex == weights.size() - 1 && outputActivationFunction != null) {
                // 出力層
                layerOutput = outputActivationFunction.activate(weightedInput); // weightedInputを配列として渡す
            } else {
                // 隠れ層
                layerOutput = new double[weightedInput.length];
                for (int i = 0; i < weightedInput.length; i++) {
                    layerOutput[i] = activationFunction.activate(weightedInput[i]);
                }
            }

            layerOutputs.add(layerOutput);
            currentInputs = layerOutput;
        }
        return currentInputs; // 出力層
    }


    // 誤差逆伝播
    public void backPropagation(double[] inputs, double[] targets) {
        t++; // タイムステップをインクリメント
        List<double[]> layerInputs = new ArrayList<>(); // 各層への入力値を保存
        List<double[]> layerOutputs = new ArrayList<>(); // 各層の出力値を保存
        List<double[]> weightedInputs = new ArrayList<>(); // 各層の線形結合の値を保存 (活性化関数適用前)

        layerInputs.add(inputs);
        layerOutputs.add(inputs);
        double[] currentInput = inputs;

        // 順伝播計算と各層の入力、出力、線形結合を保存
        for (int layerIndex = 0; layerIndex < weights.size(); layerIndex++) {
            double[][] weightMatrix = weights.get(layerIndex);
            double[] biasVector = biases.get(layerIndex);
            double[] weightedInput = new double[weightMatrix[0].length];
            double[] layerOutput = new double[weightMatrix[0].length];

            // 線形結合
            for (int j = 0; j < weightMatrix[0].length; j++) {
                double sum = 0;
                for (int i = 0; i < weightMatrix.length; i++) {
                    sum += currentInput[i] * weightMatrix[i][j];
                }
                sum += biasVector[j];
                weightedInput[j] = sum;
            }

            weightedInputs.add(weightedInput); // 線形結合の値を保存

            // 活性化関数
            if (layerIndex == weights.size() - 1 && outputActivationFunction != null) {
                layerOutput = outputActivationFunction.activate(weightedInput);
            } else {
                for (int j = 0; j < weightedInput.length; j++) {
                    layerOutput[j] = activationFunction.activate(weightedInput[j]);
                }
            }
            layerInputs.add(currentInput);
            layerOutputs.add(layerOutput);
            currentInput = layerOutput;
        }


        // 出力層の誤差を計算
        double[] outputLayerOutput = layerOutputs.get(layerOutputs.size() - 1);
        double[] outputLayerError = new double[outputLayerOutput.length];
        for (int i = 0; i < outputLayerOutput.length; i++) {
            outputLayerError[i] = outputLayerOutput[i] - targets[i]; // 交差エントロピー誤差 + Softmax の微分は prediction - target
        }

        List<double[]> weightGradients = new ArrayList<>();
        List<double[]> biasGradients = new ArrayList<>();
        double[] currentError = outputLayerError;


        // 逆伝播と勾配計算
        for (int layerIndex = weights.size() - 1; layerIndex >= 0; layerIndex--) {
            double[] prevLayerOutput = layerInputs.get(layerIndex); // layerInputsから入力値を取得
            double[][] weightMatrix = weights.get(layerIndex);
            double[] weightedInput = weightedInputs.get(layerIndex);

            double[] weightGradient = new double[weightMatrix.length * weightMatrix[0].length];
            double[] biasGradient = new double[biases.get(layerIndex).length];
            double[] nextLayerError = new double[prevLayerOutput.length];


            for (int j = 0; j < weightMatrix[0].length; j++) {
                double activationDerivative = (layerIndex == weights.size() - 1 && outputActivationFunction != null) ?
                        1.0 : activationFunction.derivative(weightedInput[j]);
                double delta = currentError[j] * activationDerivative;
                biasGradient[j] = delta;

                for (int i = 0; i < weightMatrix.length; i++) {
                    weightGradient[j * weightMatrix.length + i] = delta * prevLayerOutput[i];
                    nextLayerError[i] += weightMatrix[i][j] * delta;
                }
            }
            weightGradients.add(0, weightGradient);
            biasGradients.add(0, biasGradient);
            currentError = nextLayerError;
        }
        updateWeightsAndBiases(weightGradients, biasGradients);

    }


    // Adam Optimizerによる重みとバイアスの更新 (変更なし)
    private void updateWeightsAndBiases(List<double[]> weightGradients, List<double[]> biasGradients) {
        for (int layerIndex = 0; layerIndex < weights.size(); layerIndex++) {
            double[][] weightMatrix = weights.get(layerIndex);
            double[] biasVector = biases.get(layerIndex);
            double[] weightGradient = weightGradients.get(layerIndex);
            double[] biasGradient = biasGradients.get(layerIndex);
            double[] m_weight = m_weights.get(layerIndex);
            double[] v_weight = v_weights.get(layerIndex);
            double[] m_bias = m_biases.get(layerIndex);
            double[] v_bias = v_biases.get(layerIndex);


            // 重みの更新
            for (int j = 0; j < weightMatrix[0].length; j++) {
                for (int i = 0; i < weightMatrix.length; i++) {
                    int index = j * weightMatrix.length + i;
                    double currentWeightGradient = weightGradient[index] + l2Regularization * weightMatrix[i][j];
                    m_weight[index] = beta1 * m_weight[index] + (1 - beta1) * currentWeightGradient;
                    v_weight[index] = beta2 * v_weight[index] + (1 - beta2) * Math.pow(currentWeightGradient, 2);
                    double m_hat = m_weight[index] / (1 - Math.pow(beta1, t));
                    double v_hat = v_weight[index] / (1 - Math.pow(beta2, t));
                    weightMatrix[i][j] -= learningRate * m_hat / (Math.sqrt(v_hat) + epsilon);
                }
            }


            // バイアスの更新
            for (int j = 0; j < biasVector.length; j++) {
                double currentBiasGradient = biasGradient[j];
                m_bias[j] = beta1 * m_bias[j] + (1 - beta1) * currentBiasGradient;
                v_bias[j] = beta2 * v_bias[j] + (1 - beta2) * Math.pow(currentBiasGradient, 2);
                double m_hat = m_bias[j] / (1 - Math.pow(beta1, t));
                double v_hat = v_bias[j] / (1 - Math.pow(beta2, t));
                biasVector[j] -= learningRate * m_hat / (Math.sqrt(v_hat) + epsilon);
            }
        }
    }


    // 予測
    public double[] predict(double[] inputs) {
        return forwardPropagation(inputs);
    }


    // 学習
    public void train(List<double[]> inputData, List<double[]> targetData, int epochs, int batchSize, boolean useSMOTE) {
        if (inputData.size() != targetData.size()) {
            throw new IllegalArgumentException("入力データとターゲットデータのサイズが一致しません。");
        }

        List<double[]> processedInputData = new ArrayList<>(inputData);
        List<double[]> processedTargetData = new ArrayList<>(targetData);

        if (useSMOTE) {
            System.out.println("Applying SMOTE...");
            Map<Integer, List<Integer>> classIndices = getClassIndices(processedTargetData);
            if (classIndices.size() > 1) { // SMOTEは少なくとも2クラス分類を想定
                processedInputData = new ArrayList<>(processedInputData);
                processedTargetData = new ArrayList<>(processedTargetData);
                applySMOTE(processedInputData, processedTargetData, classIndices);
                System.out.println("SMOTE applied. New dataset size: " + processedInputData.size());
            } else {
                System.out.println("SMOTE not applied because only one class detected or not applicable.");
            }
        }


        int numSamples = processedInputData.size();
        Random random = new Random();

        for (int epoch = 0; epoch < epochs; epoch++) {
            // シャッフル
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < numSamples; i++) {
                indices.add(i);
            }
            java.util.Collections.shuffle(indices, random);

            double epochCorrectPredictions = 0; // エポックごとの正解数をカウント

            // ミニバッチ学習
            for (int i = 0; i < numSamples; i += batchSize) {
                List<double[]> batchInputs = new ArrayList<>();
                List<double[]> batchTargets = new ArrayList<>();

                for (int j = i; j < Math.min(i + batchSize, numSamples); j++) {
                    int index = indices.get(j);
                    batchInputs.add(processedInputData.get(index));
                    batchTargets.add(processedTargetData.get(index));
                }

                // バッチごとの誤差逆伝播
                for (int j = 0; j < batchInputs.size(); j++) {
                    backPropagation(batchInputs.get(j), batchTargets.get(j));
                    double[] prediction = predict(batchInputs.get(j));
                    if (isCorrectPrediction(prediction, batchTargets.get(j))) { // 正解判定
                        epochCorrectPredictions++;
                    }
                }
            }

            // エポックごとの損失と正解率を計算
            double crossEntropyError = calculateCrossEntropyError(processedInputData, processedTargetData);
            double accuracy = epochCorrectPredictions / numSamples;
            System.out.println("Epoch " + (epoch + 1) + ", Cross-Entropy Error: " + crossEntropyError + ", Accuracy: " + String.format("%.2f%%", accuracy * 100));
        }
    }

    // 予測が正解かどうかを判定する
    private boolean isCorrectPrediction(double[] prediction, double[] target) {
        int predictedClass = getPredictedClass(prediction);
        int targetClass = getPredictedClass(target);
        return predictedClass == targetClass;
    }


    // 交差エントロピー誤差
    public double calculateCrossEntropyError(List<double[]> inputData, List<double[]> targetData) {
        double crossEntropyError = 0.0;
        for (int i = 0; i < inputData.size(); i++) {
            double[] prediction = predict(inputData.get(i));
            double[] target = targetData.get(i);
            for(int j=0; j<prediction.length; ++j) {
                crossEntropyError -= target[j] * Math.log(prediction[j] + 1e-7);
            }
        }
        return crossEntropyError / inputData.size();
    }


    // SMOTE
    private void applySMOTE(List<double[]> inputData, List<double[]> targetData, Map<Integer, List<Integer>> classIndices) {
        int minorityClass = findMinorityClass(classIndices);
        if (minorityClass == -1) return;

        int k = 5;
        int numSyntheticSamples = calculateNumSyntheticSamples(classIndices, minorityClass);

        if (numSyntheticSamples <= 0) return;

        Random random = new Random();
        List<double[]> minorityInputData = getSamplesByClass(inputData, classIndices, minorityClass);

        for (int i = 0; i < numSyntheticSamples; i++) {
            int randomIndex = random.nextInt(minorityInputData.size());
            double[] baseSample = minorityInputData.get(randomIndex);
            List<double[]> neighbors = findNearestNeighbors(baseSample, minorityInputData, k);

            if (!neighbors.isEmpty()) {
                double[] neighborSample = neighbors.get(random.nextInt(neighbors.size()));
                double[] syntheticSample = generateSyntheticSample(baseSample, neighborSample, random);
                inputData.add(syntheticSample);
                targetData.add(targetData.get(classIndices.get(minorityClass).get(randomIndex)));
            }
        }
    }

    // クラスごとのサンプル数を計算
    private Map<Integer, List<Integer>> getClassIndices(List<double[]> targetData) {
        Map<Integer, List<Integer>> classIndices = new HashMap<>();
        for (int i = 0; i < targetData.size(); i++) {
            int classLabel = getClassLabel(targetData.get(i));
            classIndices.computeIfAbsent(classLabel, k -> new ArrayList<>()).add(i);
        }
        return classIndices;
    }

    // ターゲットデータからクラスラベルを取得 (one-hot encodingを想定) 
    private int getClassLabel(double[] target) {
        for (int i = 0; i < target.length; i++) {
            if (target[i] == 1.0) {
                return i;
            }
        }
        return -1; // クラスラベルが見つからない場合
    }


    // マイノリティクラスを特定 (サンプル数が最も少ないクラス) 
    private int findMinorityClass(Map<Integer, List<Integer>> classIndices) {
        if (classIndices.isEmpty()) return -1;

        int minorityClass = -1;
        int minSamples = Integer.MAX_VALUE;

        for (Map.Entry<Integer, List<Integer>> entry : classIndices.entrySet()) {
            if (entry.getValue().size() < minSamples) {
                minSamples = entry.getValue().size();
                minorityClass = entry.getKey();
            }
        }
        return minorityClass;
    }

    // 生成する合成サンプル数を計算 (最大クラスのサンプル数とマイノリティクラスのサンプル数の差) 
    private int calculateNumSyntheticSamples(Map<Integer, List<Integer>> classIndices, int minorityClass) {
        if (classIndices.isEmpty() || minorityClass == -1) return 0;

        int maxSamples = 0;
        int minoritySamples = classIndices.get(minorityClass).size();

        for (Map.Entry<Integer, List<Integer>> entry : classIndices.entrySet()) {
            if (entry.getValue().size() > maxSamples) {
                maxSamples = entry.getValue().size();
            }
        }
        return Math.max(0, maxSamples - minoritySamples); // 差が負にならないように
    }


    // 特定のクラスのサンプルを取得 
    private List<double[]> getSamplesByClass(List<double[]> inputData, Map<Integer, List<Integer>> classIndices, int classLabel) {
        List<double[]> samples = new ArrayList<>();
        if (classIndices.containsKey(classLabel)) {
            for (int index : classIndices.get(classLabel)) {
                samples.add(inputData.get(index));
            }
        }
        return samples;
    }


    // ユークリッド距離を計算 
    private double euclideanDistance(double[] sample1, double[] sample2) {
        if (sample1.length != sample2.length) {
            throw new IllegalArgumentException("サンプル特徴量数が異なります。");
        }
        double sumSquaredDiff = 0;
        for (int i = 0; i < sample1.length; i++) {
            sumSquaredDiff += Math.pow(sample1[i] - sample2[i], 2);
        }
        return Math.sqrt(sumSquaredDiff);
    }

    // 最近傍探索 (k-NN) 
    private List<double[]> findNearestNeighbors(double[] baseSample, List<double[]> sampleList, int k) {
        List<NeighborDistance> neighborDistances = new ArrayList<>();
        for (double[] sample : sampleList) {
            if (sample != baseSample) {
                double distance = euclideanDistance(baseSample, sample);
                neighborDistances.add(new NeighborDistance(sample, distance));
            }
        }

        neighborDistances.sort(Comparator.comparingDouble(nd -> nd.distance));

        List<double[]> neighbors = new ArrayList<>();
        int count = 0;
        for (NeighborDistance nd : neighborDistances) {
            if (count < k && count < neighborDistances.size()) {
                neighbors.add(nd.neighbor);
                count++;
            } else {
                break;
            }
        }
        return neighbors;
    }

    // 合成サンプルを生成
    private double[] generateSyntheticSample(double[] baseSample, double[] neighborSample, Random random) {
        if (baseSample.length != neighborSample.length) {
            throw new IllegalArgumentException("サンプル特徴量数が異なります。");
        }
        double[] syntheticSample = new double[baseSample.length];
        for (int i = 0; i < baseSample.length; i++) {
            double diff = neighborSample[i] - baseSample[i];
            double gap = random.nextDouble();
            syntheticSample[i] = baseSample[i] + gap * diff;
        }
        return syntheticSample;
    }

    // 予測クラスを取得 (最も確率の高いクラス) 
    public int getPredictedClass(double[] prediction) {
        int predictedClass = 0;
        double maxProbability = -1.0;
        for (int i = 0; i < prediction.length; i++) {
            if (prediction[i] > maxProbability) {
                maxProbability = prediction[i];
                predictedClass = i;
            }
        }
        return predictedClass;
    }
}