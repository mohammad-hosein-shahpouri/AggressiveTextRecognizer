using Microsoft.ML.Data;

namespace AggressiveTextRecognizer.Models;

public class Output
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    public float Probability { get; set; }
}