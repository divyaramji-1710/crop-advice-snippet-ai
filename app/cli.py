"""Command-line interface for the crop-text GPT."""
import argparse

from app.generation import SnippetGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="crop-advice-snippet-ai",
        description=(
            "Generate experimental tomato-growing text "
            "with a GPT trained from scratch."
        ),
    )
    commands = parser.add_subparsers(
        dest="command", required=True
    )

    commands.add_parser(
        "hello", help="check that the application runs"
    )

    generate = commands.add_parser(
        "generate",
        help="continue a crop or problem prompt",
    )
    generate.add_argument(
        "prompt",
        nargs="?",
        help="crop or problem text; omitted for an interactive prompt",
    )
    generate.add_argument(
        "--checkpoint",
        default="artifacts/best-model/best.pt",
    )
    generate.add_argument("--max-new-tokens", type=int, default=60)
    generate.add_argument("--temperature", type=float, default=0.6)
    generate.add_argument("--top-k", type=int, default=10)
    generate.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "hello":
        print(
            "Crop Advice Snippet AI is ready. "
            "Open README.md for your milestones."
        )
        return

    prompt = args.prompt
    if prompt is None:
        prompt = input("Enter a crop or problem: ")

    if not prompt.strip():
        parser.error("Enter a non-empty prompt")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    try:
        generator = SnippetGenerator(args.checkpoint)
        continuation = generator.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))

    print(
        "\nExperimental model output from a historical corpus."
        "\nNot verified farming advice."
    )
    print(f"\nPrompt: {prompt}")
    print(f"Continuation: {continuation}")


if __name__ == "__main__":
    main()