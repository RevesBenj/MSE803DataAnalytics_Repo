# Ethical and Culturally Relevant Risk Management

## Running App Data Analysis Project with Recommendation Feature

### Purpose

The Running App analyses users' fitness data to provide personalised exercise recommendations. However, recommending the same intensive running program to all users may cause injuries, unfair outcomes, and loss of user trust. The system should therefore follow ethical principles, protect privacy, and respect cultural diversity throughout the data lifecycle.

## 1. Ethical Implications

The app should only collect data that is necessary for improving recommendations. Users must provide informed consent before their data is collected or analysed. Recommendations should be transparent, explainable, and should not replace professional medical advice. The system should avoid discrimination and provide fair recommendations for all users regardless of age, gender, ethnicity, disability, or fitness level.

## 2. Privacy Concerns and Risk Mitigation

The app collects sensitive information such as GPS location, heart rate, age, weight, and exercise history. These data may expose users to privacy risks if they are leaked or misused.

Risk mitigation includes:

* Collect only necessary data (data minimisation).
* Obtain informed user consent.
* Encrypt and securely store all personal data.
* Use anonymisation or pseudonymisation where possible.
* Restrict data access to authorised personnel.
* Allow users to view, update, download, or delete their personal data.

## 3. Cultural Relevance

The recommendation system should avoid a one-size-fits-all approach. It should consider different cultures, languages, disabilities, family responsibilities, religious practices, financial situations, and access to safe exercise environments.

In New Zealand, the project should comply with the **Privacy Act 2020**, respect **Te Tiriti o Waitangi**, and recognise **Māori Data Sovereignty**, ensuring Māori data is governed with Māori participation and benefits Māori communities.

## 4. Dataset Analysis

Before building the recommendation model, the dataset should be analysed to identify:

* Missing values across demographic groups.
* Sampling bias and under-represented users.
* Recommendation performance across age, gender, ethnicity, disability, and fitness levels.
* Fairness metrics to ensure similar model performance for all user groups.

If bias is detected, additional representative data should be collected and the model retrained before deployment.

## 5. Data Collection Guidelines

| **Collect**             | **Do Not Collect**                   |
| ----------------------- | ------------------------------------ |
| Age range               | Bank details                         |
| Fitness level           | Passwords                            |
| Running history         | Irrelevant personal files            |
| Health goals            | Unnecessary contacts                 |
| Heart rate (optional)   | Data unrelated to fitness            |
| GPS location (optional) | Personal information without consent |

## 6. Fairness and Ethical Guidelines

The recommendation model should be tested regularly for fairness and accuracy across different demographic groups. Performance differences should be investigated and corrected. Users should receive clear explanations of recommendations, have the ability to opt out of personalised recommendations, and request human review when needed.

## Conclusion

An ethical Running App should provide safe, personalised, and culturally appropriate recommendations rather than giving intensive exercise advice to every user. By applying privacy-by-design, informed consent, fairness testing, cultural inclusion, continuous monitoring, and compliance with New Zealand ethical principles, the project can reduce risks, improve trust, and deliver responsible data-driven recommendations.
