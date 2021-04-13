using AggressiveTextRecognizer.Models;
using Microsoft.Extensions.ML;

namespace AggressiveTextRecognizer
{
    public class TextRecognizer : ITextRecognizer
    {
        private PredictionEnginePool<Input, Output> PredictionEnginePool;

        public TextRecognizer(PredictionEnginePool<Input, Output> PredictionEnginePool) =>
            this.PredictionEnginePool = PredictionEnginePool;

        public Output IsAggressive(string text) =>
            PredictionEnginePool.Predict(new Input()
            {
                Comment = text
            });
    }
}
