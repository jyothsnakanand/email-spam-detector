"""Generate synthetic labeled email data for training."""

import random
from pathlib import Path
from typing import Any

import pandas as pd

from spam_detector.config import settings

# Sample spam keywords and phrases
SPAM_KEYWORDS = [
    "free",
    "winner",
    "congratulations",
    "claim",
    "prize",
    "click here",
    "limited time",
    "act now",
    "urgent",
    "offer",
    "discount",
    "buy now",
    "credit card",
    "loan",
    "casino",
    "viagra",
    "pharmacy",
    "weight loss",
    "make money",
    "work from home",
    "guaranteed",
    "risk-free",
    "no obligation",
    "cash bonus",
    "dear friend",
]

# Sample legitimate email phrases
LEGITIMATE_PHRASES = [
    "meeting scheduled",
    "project update",
    "please review",
    "quarterly report",
    "team lunch",
    "calendar invite",
    "code review",
    "documentation",
    "thanks for",
    "attached is",
    "following up",
    "as discussed",
    "let me know",
    "best regards",
    "looking forward",
    "proposal",
    "invoice",
    "delivery confirmation",
    "support ticket",
    "customer feedback",
]


def generate_spam_email() -> str:
    """Generate a synthetic spam email."""
    templates = [
        f"CONGRATULATIONS! You've won a {random.choice(['prize', 'lottery', 'jackpot'])}! "
        f"{random.choice(['Click here', 'Act now', 'Claim immediately'])} to claim your "
        f"{random.choice(['cash', 'reward', 'bonus'])}!",
        f"URGENT: {random.choice(['Limited time offer', 'Special discount', 'Exclusive deal'])}! "
        f"Get {random.choice(['50%', '70%', '90%'])} off {random.choice(['viagra', 'weight loss pills', 'casino credits'])}. "
        f"{random.choice(['Buy now', 'Order today', 'Click here'])}!",
        f"Dear friend, make ${random.randint(1000, 10000)} {random.choice(['per day', 'per week', 'from home'])}! "
        f"{random.choice(['No experience needed', 'Risk-free', 'Guaranteed income'])}. "
        f"{random.choice(['Apply now', 'Start today', 'Click to learn more'])}!",
        f"ALERT: Your {random.choice(['credit card', 'bank account', 'loan'])} has been "
        f"{random.choice(['approved', 'upgraded', 'selected'])}! "
        f"Click here for {random.choice(['instant cash', 'free money', 'bonus credit'])}.",
    ]
    return random.choice(templates)


def generate_legitimate_email() -> str:
    """Generate a synthetic legitimate email."""
    templates = [
        f"Hi team, {random.choice(['meeting scheduled', 'project update', 'quick reminder'])} "
        f"for {random.choice(['tomorrow', 'next week', 'this afternoon'])}. "
        f"{random.choice(['Please review', 'Looking forward', 'See you then'])}.",
        f"Hello, {random.choice(['attached is', 'please find', 'here is'])} the "
        f"{random.choice(['quarterly report', 'documentation', 'proposal', 'invoice'])} "
        f"we discussed. {random.choice(['Let me know if you have questions', 'Thanks', 'Best regards'])}.",
        f"Following up on {random.choice(['our conversation', 'the code review', 'your request'])}. "
        f"{random.choice(['The changes look good', 'Everything is approved', 'Ready to proceed'])}. "
        f"{random.choice(['Thanks for your help', 'Let me know', 'Best'])}.",
        f"Hi, {random.choice(['calendar invite sent', 'delivery confirmation', 'support ticket updated'])} "
        f"for {random.choice(['your order', 'the meeting', 'case #' + str(random.randint(1000, 9999))])}. "
        f"{random.choice(['Thanks', 'Regards', 'Best'])}.",
    ]
    return random.choice(templates)


def generate_dataset(
    num_samples: int = 5000, spam_ratio: float = 0.4, output_path: Path | None = None
) -> pd.DataFrame:
    """
    Generate a labeled dataset of emails.

    Args:
        num_samples: Total number of samples to generate
        spam_ratio: Ratio of spam emails (0.0 to 1.0)
        output_path: Path to save the dataset CSV file

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    num_spam = int(num_samples * spam_ratio)
    num_legitimate = num_samples - num_spam

    data: list[dict[str, Any]] = []

    # Generate spam emails
    for _ in range(num_spam):
        data.append({"text": generate_spam_email(), "label": "spam"})

    # Generate legitimate emails
    for _ in range(num_legitimate):
        data.append({"text": generate_legitimate_email(), "label": "ham"})

    # Shuffle the dataset
    random.shuffle(data)

    df = pd.DataFrame(data)

    # Save to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")

    return df


def main() -> None:
    """Generate dataset using settings from environment."""
    print(f"Generating {settings.generated_data_size} samples...")
    print(f"Spam ratio: {settings.spam_ratio}")

    df = generate_dataset(
        num_samples=settings.generated_data_size,
        spam_ratio=settings.spam_ratio,
        output_path=settings.train_data_path,
    )

    print("\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()
