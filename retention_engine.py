def get_recommendations(reasons):

    recommendations = []

    for reason in reasons:

        if reason == "Low tenure":
            recommendations.append(
                "Provide onboarding support and regular follow-ups."
            )

        elif reason == "High monthly charges":
            recommendations.append(
                "Offer loyalty discounts or lower-cost plans."
            )

        elif reason == "Month-to-month contract":
            recommendations.append(
                "Offer incentives for one-year or two-year contracts."
            )

        elif reason == "Electronic check payment method":
            recommendations.append(
                "Encourage automatic payment methods."
            )

        elif reason == "Fiber optic internet service":
            recommendations.append(
                "Monitor service quality and customer satisfaction."
            )

    return recommendations