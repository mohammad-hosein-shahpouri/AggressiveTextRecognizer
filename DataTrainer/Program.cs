using AggressiveTextRecognizer.Models;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace DataTrainer
{
    class Program
    {
        public static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(0);

            // Load Data
            Console.WriteLine("Loading Data...");
            var createInputFile = @"Data\prepairedInput.tsv";
            CreatePrepairedDataFile(createInputFile, true);

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<Input>(
                path: createInputFile,
                hasHeader: true,
                separatorChar: '\t',
                allowQuoting: true);

            var inputDataSplit = mlContext.Data.TrainTestSplit
                (trainingDataView, testFraction: 0.25, seed: 0);

            // Build Pipeline
            Console.WriteLine("Building Pipeline...");

            var inputDataPrepairer = mlContext.Transforms.Text
                .FeaturizeText("Features", "Comment")
                .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Fit(inputDataSplit.TrainSet);

            var trainer = mlContext.BinaryClassification
                .Trainers.LbfgsLogisticRegression();

            // Fit The Model
            Console.WriteLine("Fitting The Model...");

            var transformedData = inputDataPrepairer.Transform(inputDataSplit.TrainSet);
            ITransformer model = trainer.Fit(transformedData);

            // Test The Model
            Console.WriteLine("Testing The Model...");

            EvaluateModel(mlContext, model, inputDataPrepairer.Transform(inputDataSplit.TestSet));

            // Save The Model  
            Console.WriteLine("Saving The Model...");

            if (!Directory.Exists(Path.Combine("Data", "Model")))
                Directory.CreateDirectory(Path.Combine("Data", "Model"));
            var modelFile = Path.Combine("Data", "Model", "Model.zip");
            mlContext.Model.Save(model, trainingDataView.Schema, modelFile);

            var dataPrepairePipelinefile = @"Data\Model\dataPrepairePipeline.zip";
            mlContext.Model.Save(inputDataPrepairer, trainingDataView.Schema, dataPrepairePipelinefile);

            var retrainedModel = RetrainedModel(modelFile, dataPrepairePipelinefile);

            var compeleteRetrainedPipeline = inputDataPrepairer.Append(retrainedModel);

            var retarinedModelFile = @"Data\Model\AggressiveTextRecognizerData.zip";//This is The Required File for the Library
            mlContext.Model.Save(compeleteRetrainedPipeline, trainingDataView.Schema, retarinedModelFile);

            Console.WriteLine("Testing The Model for The Final Result...");
            EvaluateModel(mlContext, compeleteRetrainedPipeline, inputDataSplit.TestSet);

            Console.WriteLine($"Your File is Located at {retarinedModelFile}.");
            Console.WriteLine("Please Copy The File Where Your Web Application (.csproj) is.");
        }

        private static ITransformer RetrainedModel(string modelFile, string dataPrepairePipelinefile)
        {
            MLContext mlContext = new MLContext(0);

            ITransformer preTrainedModel = mlContext.Model.Load(modelFile, out _);
            var preTrainedModelParameters = ((ISingleFeaturePredictionTransformer
                <CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>)
                preTrainedModel).Model.SubModel;

            var dataFile = @"Data\prepairedInput.tsv";
            CreatePrepairedDataFile(dataFile, false);

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<Input>(
                path: dataFile,
                hasHeader: true,
                separatorChar: '\t',
                allowQuoting: true);

            ITransformer dataPrepairePipeline = mlContext.Model.Load(dataPrepairePipelinefile, out _);

            var newData = dataPrepairePipeline.Transform(trainingDataView);

            var retrainedModel = mlContext
                .BinaryClassification.Trainers
                    .LbfgsLogisticRegression().Fit(newData, preTrainedModelParameters);


            return retrainedModel;
        }

        private static void CreatePrepairedDataFile(string outputFile, bool onlySaveSmallSubset = false)
        {
            var annotations = File.ReadAllLines(@"Data\aggression_annotations.tsv").Skip(1);
            var aggressiveScoreMap = new Dictionary<int, List<int>>();

            //Collect all aggression ratings for each comment (revId)
            foreach (var annotation in annotations)
            {
                var parts = annotation.Split('\t');

                var revId = int.Parse(parts[0]);
                var aggressiveScore = (int)double.Parse(parts[3], CultureInfo.InvariantCulture);

                if (aggressiveScoreMap.ContainsKey(revId))
                    aggressiveScoreMap[revId].Add(aggressiveScore);
                else
                    aggressiveScoreMap[revId] = new List<int>() { aggressiveScore };
            }

            // Pair all comments with aggression score
            var allComments = File.ReadAllLines(@"Data\aggression_annotated_comments.tsv").Skip(1);

            var formattedOutput = allComments.Select(c =>
            {
                var inputLineParts = c.Split('\t');

                var commentId = int.Parse(inputLineParts[0]);

                var aggressionScores = aggressiveScoreMap[commentId];

                var aggression = aggressionScores.Average() < -0.9 ? 1 : 0;

                var comment = inputLineParts[1].Replace("NEWLINE_TOKEN", "");
                return $"{aggression}\t{comment}";
            });

            // Take the small or the big subset of the data
            var finalOutput = onlySaveSmallSubset ?
                formattedOutput.Take(3000).ToList() :
                formattedOutput.Skip(3000).ToList();

            finalOutput.Insert(0, "IsAggressive\tComment");

            //Write the new file to use as ML.Net input
            File.WriteAllLines(outputFile, finalOutput);
        }

        private static void EvaluateModel(MLContext mlContext, ITransformer trainedData, IDataView testData)
        {
            var predictedData = trainedData.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictedData);
            Console.WriteLine($"Accuracy:{(metrics.Accuracy * 100):###.###}");
        }

    }
}
