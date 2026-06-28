# Hotel Booking Data Analysis - Output Summary

## Dataset Size
- Raw rows: 119,390
- Clean rows: 87,228
- Columns after preprocessing: 41

## Cleaning Completed
- Removed exact duplicates.
- Filled missing company and agent as 0.
- Filled missing country as Unknown.
- Filled missing children as 0.
- Removed bookings with zero total guests.
- Removed invalid or extreme ADR values.
- Created arrival_date, total_guests, total_nights, revenue_estimate, and peak season features.

## Key Results
- Peak booking month: 2017-05 with 4,567 bookings.
- Most common reserved room type: A with 56,434 bookings.
- Cancellation rate by hotel:
hotel
City Hotel      30.10
Resort Hotel    23.48

## Best-Performing Approach
Best model: Random Forest

Metrics:
- Accuracy: 0.7709
- Precision: 0.5552
- Recall: 0.8427
- F1 Score: 0.6694
- ROC-AUC: 0.8826

## Business Insights
1. High lead time is linked with higher cancellation risk.
2. Some hotel types and customer types have different cancellation behavior.
3. ADR changes across months, supporting dynamic pricing.
4. Popular room types should be prioritized in inventory and promotion.
5. Cancellation prediction can support better staffing, overbooking control, and revenue planning.

## Saved Outputs
- Tables: output/tables/
- Figures: output/figures/
- Best model: output/models/best_cancellation_model.joblib
