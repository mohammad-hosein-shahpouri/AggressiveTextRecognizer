using AggressiveTextRecognizer.Models;

namespace AggressiveTextRecognizer
{
    public interface ITextRecognizer
    {
        Output IsAggressive(string text);
    }
}
