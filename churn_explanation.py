def explain_churn(customer):

    reasons = []

    if customer['tenure'] < 12:
        reasons.append("Low tenure")

    if customer['MonthlyCharges'] > 80:
        reasons.append("High monthly charges")

    if customer['Contract_One year'] == 0 and customer['Contract_Two year'] == 0:
        reasons.append("Month-to-month contract")

    if customer['PaymentMethod_Electronic check'] == 1:
        reasons.append("Electronic check payment method")

    if customer['InternetService_Fiber optic'] == 1:
        reasons.append("Fiber optic internet service")

    return reasons