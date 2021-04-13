using Microsoft.ML.Data;

namespace AggressiveTextRecognizer.Models
{
    public class Input
    {
        [LoadColumn(1)]
        public string Comment { get; set; }
        [LoadColumn(0), ColumnName("Label")]
        public bool IsAggressive { get; set; }
    }
}
