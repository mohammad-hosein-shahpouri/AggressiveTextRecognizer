using AggressiveTextRecognizer.Models;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;

namespace AggressiveTextRecognizer.Services
{
    public static class AggressiveTextRecognizer
    {
        private static readonly string modelFile = @"AggressiveTextRecognizerData.zip";

        public static IServiceCollection AddAggressiveTextRecognizer(this IServiceCollection services)
        {
            services.AddSingleton<ITextRecognizer, TextRecognizer>();

            services.AddPredictionEnginePool<Input, Output>()
                .FromFile(filePath: modelFile, watchForChanges: true);

            return services;
        }
    }
}
