using AggressiveTextRecognizer;
using AggressiveTextRecognizer.Models;
using Microsoft.AspNetCore.Mvc;

namespace TestOnWeb.Controllers;

[ApiController]
[Route("[controller]")]
public class IsAggressiveController : ControllerBase
{
    private readonly ITextRecognizer textRecognizer;

    public IsAggressiveController(ITextRecognizer textRecognizer)
        => this.textRecognizer = textRecognizer;

    [HttpGet, HttpPost]
    public Output Index([FromForm, FromBody] string text)
        => text == null ? null : textRecognizer.IsAggressive(text);
}