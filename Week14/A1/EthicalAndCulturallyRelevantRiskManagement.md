# Ethical and Culturally Relevant Risk Management

## Running App Data Analysis and Recommendation Feature

### 1. Purpose and Context

The Running App analyses users’ fitness activities and provides personalised recommendations. A major ethical risk happens when the app recommends intensive exercise to every user. Users have different ages, fitness levels, disabilities, health conditions, cultures, locations, and personal goals. Therefore, the recommendations must be safe, fair, explainable, and culturally respectful.

### 2. Main Ethical Risks

**Health and safety:** Unsuitable recommendations may cause injury, exhaustion, or worsen an existing health condition.

**Privacy:** GPS routes, heart rate, age, weight, running habits, and health details are sensitive information. Location data may also reveal a user’s home, workplace, or daily routine.

**Lack of informed consent:** Users may not clearly understand how their information is collected, analysed, stored, shared, or used by the recommendation model.

**Bias and unfairness:** If the dataset mainly represents young, healthy, and highly active users, older people, beginners, people with disabilities, and other under-represented groups may receive inaccurate advice.

**Lack of transparency:** Users may believe that automated recommendations are medical advice, even though they are only general fitness suggestions.

### 3. Cultural Relevance Risks

The app must avoid a one-size-fits-all approach. Recommendations should consider language, disability, family responsibilities, financial limitations, religious or cultural practices, and access to safe running locations.

In New Zealand, Māori data should be managed with Māori participation, respect, shared benefit, and appropriate governance. The project should consider the Privacy Act 2020, Te Tiriti o Waitangi principles, and Māori data sovereignty.

### 4. Risk Management Controls

1. Collect only the information required for analysis and recommendations.
2. Obtain clear informed consent and explain the purpose of each data field.
3. Make GPS, demographic, and health information optional where possible.
4. Encrypt data, restrict staff access, and use pseudonymised user identifiers.
5. Allow users to access, correct, download, and delete their information.
6. Ask users about fitness level, goals, limitations, accessibility needs, and preferred exercise intensity.
7. Provide beginner, moderate, and advanced recommendations instead of intensive advice for everyone.
8. Explain why each recommendation was generated.
9. Include safety warnings and advise users to consult a health professional when appropriate.
10. Test model accuracy and recommendation outcomes across age, gender, ethnicity, disability, and fitness groups.
11. Provide feedback, opt-out, and human review options.
12. Regularly review the system for bias, unsafe recommendations, data breaches, model drift, and user complaints.

### 5. Fairness and Monitoring

Fairness should be evaluated before and after deployment. The project team should compare recommendation acceptance, completion rates, model errors, complaints, safety incidents, and user satisfaction across different demographic groups.

Large performance differences between groups should be investigated and corrected. A responsible staff member should be assigned to manage privacy, fairness, cultural consultation, and incident response.

### 6. Conclusion

The Running App should provide supportive and personalised guidance rather than giving intensive exercise advice to every user. Privacy-by-design, informed consent, fairness testing, cultural consultation, transparency, human oversight, and continuous monitoring will reduce harm and improve user trust.
