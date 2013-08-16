import static liblbfgs.LBFGS.LBFGS_LINESEARCH_BACKTRACKING;

import java.io.FileNotFoundException;
import java.util.List;

import liblbfgs.LBFGS;
import liblbfgs.LBFGSClient;
import liblbfgs.lbfgs_parameter_t;

/**
 * @TODO simple introduction
 * 
 *       <p>
 *       detailed comment
 * @author wangqifeng
 * @date 2013 8 15 10:32:38
 * @since
 */

public class LogisticRegressionUseLBFGS implements LBFGSClient
{
    List<Instance> instances;

    public LogisticRegressionUseLBFGS(List<Instance> instances)
    {
        this.instances = instances;
    }

    public double evaluate(Object object, double[] x, double[] g, int n,
            double step)
    {
        // x 相当与逻辑回归里的weight，需要计算新的f(x)值以及梯度
        // 这里f(x)为cost function f(x) = 1/2m * ∑(WX-Y)^2
        double fx = 0.;
        int m = this.instances.size();
        double deta = 0.;

        for (int i = 0; i < m; i++)
        {
            double[] realX = instances.get(i).getX();
            deta = classify(realX, x, n) - instances.get(i).getLabel();
            fx += deta * deta;
            for (int j = 0; j < n; j++)
            {
                g[j] += deta * realX[j];
            }
        }
        fx = fx / (2 * m);
        for (int j = 0; j < n; j++)
        {
            g[j] = g[j] / m;
        }

        // g[i] =
        return fx;
    }

    public int progress(Object instance, double[] x, double[] g, double[] fx,
            double xnorm, double gnorm, double step, int n, int k, int ls)
    {
        System.out.print(String.format("Iteration %d:\n", k));
        System.out.print(String.format(
                "  fx = %f, x[0] = %f, x[1] = %f, x[2]=%f\n", fx[0], x[0],
                x[1], x[2]));
        System.out.print(String.format("  xnorm = %f, gnorm = %f, step = %f\n",
                xnorm, gnorm, step));
        System.out.println();
        return 0;
    }

    public static double classify(double[] x, double[] weights, int n)
    {
        double logit = .0;
        for (int i = 0; i < n; i++)
        {
            logit += weights[i] * x[i];
        }
        return sigmoid(logit);
    }

    public static double sigmoid(double z)
    {
        return 1 / (1 + Math.exp(-z));
    }

    public double[] train()
    {
        lbfgs_parameter_t param = LBFGS._defparam.clone();

        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;

        // 构造变量的初值

        double fx = 0;

        int n = instances.get(0).getX().length;

        double[] w = new double[n];
        for (int i = 0; i < n; i++)
        {
            w[i] = 0.5;
        }

        int ret = LBFGS.lbfgs(n, w, fx, this, this, null, param);

        return w;

    }

    public static void main(String... args) throws FileNotFoundException
    {
        List<Instance> instances = DataSet.readDataSet("testSet.txt");
        List<Instance> instances1 = DataSet.readDataSet("testSet1.txt");
        LogisticRegressionUseLBFGS logistic = new LogisticRegressionUseLBFGS(
                instances);

        double[] w = logistic.train();
        int flasecount = 0;
        for (Instance test : instances)
        {
            double probx = logistic.classify(test.getX(), w, w.length);
            double y = test.getLabel();
            boolean correct = true;
            if ((probx >= 0.5 && y == 0) ||  (probx < 0.5 && y == 1))
            {
                correct = false;
                flasecount++;
            }
            if(!correct)
            {
            System.out.println("prob(x) = " + probx + " and real is: "
                    + test.getLabel() +" and result is:"+correct);
            }
            else
            {
                System.out.println("prob(x) = " + probx + " and real is: "
                        + test.getLabel());  
            }
            
        }
        System.out.println("total line:"+instances.size()+", and incorrect count:"+flasecount);

    }

}
