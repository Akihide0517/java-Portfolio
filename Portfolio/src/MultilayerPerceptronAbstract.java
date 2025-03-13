import java.util.List;
//人工知能構築に必要な抽象的要素
public abstract class MultilayerPerceptronAbstract {
	int[] hiddenLayerSizes; // 各隠れ層のニューロン数。{4, 3} のように指定すれば、1層目の隠れ層は4ニューロン、2層目の隠れ層は3ニューロンとなる
	List<double[][]> weights; // 重み行列
    List<double[]> biases; // バイアスベクトル
	
    public MultilayerPerceptronAbstract() {
    	//ここでは継承先でパラメータの初期化を行う
    	System.out.println("起動しました");
    }
    
    //準伝搬
    public abstract double[] forwardPropagation(double[] inputs);
    
    //逆伝搬
    public abstract void backPropagation(double[] inputs, double[] targets);
    
    //予測（正確には伝搬の結果取得）
    public abstract double[] predict(double[] inputs);
    
    //学習
    public abstract void train(List<double[]> inputData, List<double[]> targetData, int epochs, int batchSize, boolean useSMOTE);
    
    //クロスエントロピー誤差による勾配取得
    public abstract double calculateCrossEntropyError(List<double[]> inputData, List<double[]> targetData);
    
    //学習結果を用いてどのラベルが０に近いか返す
    public abstract int getPredictedClass(double[] prediction);
}
