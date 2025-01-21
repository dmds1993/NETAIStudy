using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;

class Program
{
    static void Main()
    {
        string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "BoardMemberResponse.mbconfig");


        // Load the trained model
        var mlContext = new MLContext();
        ITransformer model = mlContext.Model.Load(modelPath, out var inputSchema);

        // Create a prediction engine
        var predictor = mlContext.Model.CreatePredictionEngine<EmailData, EmailPrediction>(model);
        TestModel(predictor);


    }

    public static void TestModel(PredictionEngine<EmailData, EmailPrediction> predictor)
    {
        var testSamples = new List<EmailData>
            {
                new EmailData { Text = "Can I get more information about the product?" },
                new EmailData { Text = "We are pleased to inform you that the proposal is accepted." },
                new EmailData { Text = "We regret to inform you that your request was rejected." }
            };

        foreach (var sample in testSamples)
        {
            var prediction = predictor.Predict(sample);

            // Display results
            Console.WriteLine($"prediction: {JsonConvert.SerializeObject(prediction)}");
            Console.WriteLine($"Email: {sample.Text}");
            Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
            Console.WriteLine($"Approval Confidence: {prediction.Score[0]:P2}");
            Console.WriteLine($"Rejection Confidence: {prediction.Score[1]:P2}");
            Console.WriteLine($"Question Confidence: {prediction.Score[2]:P2}");
        }
    }
}



// Email data structure
public class EmailData
{
    [LoadColumn(0)] public string Text { get; set; }
    [LoadColumn(1)] public string Label { get; set; }
}

// Prediction class to include confidence scores
public class EmailPrediction
{
    public uint PredictedLabelKey { get; set; }  // Key predicted by ML.NET
    public string PredictedLabel { get; set; }   // Mapped string value
    public float[] Score { get; set; }
}
