use chrono::{DateTime, Datelike, Timelike, Utc};

use crate::model::FraudScoreRequest;

use super::*;

impl FraudEngine {
    pub fn vectorize(
        &self,
        req: &FraudScoreRequest,
    ) -> Result<[f32; VECTOR_DIMENSIONS], FraudEngineError> {
        let requested_at = loader::parse_utc_timestamp(&req.transaction.requested_at)?;

        let amount =
            math::clamp_ratio(req.transaction.amount as f32, self.normalization.max_amount);

        let installments = math::clamp_ratio(
            req.transaction.installments as f32,
            self.normalization.max_installments,
        );

        let amount_vs_avg = math::normalize_amount_vs_avg(
            req.transaction.amount as f32,
            req.customer.avg_amount as f32,
            self.normalization.amount_vs_avg_ratio,
        );

        let hour_of_day = requested_at.hour() as f32 / 23.0;
        let day_of_week = requested_at.weekday().num_days_from_monday() as f32 / 6.0;

        let (minutes_since_last_tx, km_from_last_tx) = self.get_last_tx_data(req, requested_at)?;

        let km_from_home =
            math::clamp_ratio(req.terminal.km_from_home as f32, self.normalization.max_km);
        let tx_count_24h = math::clamp_ratio(
            req.customer.tx_count_24h as f32,
            self.normalization.max_tx_count_24h,
        );

        let is_online = math::bool_to_unit(req.terminal.is_online);
        let card_present = math::bool_to_unit(req.terminal.card_present);

        let unknown_merchant = if req
            .customer
            .known_merchants
            .iter()
            .any(|known_merchant| known_merchant == &req.merchant.id)
        {
            0.0
        } else {
            1.0
        };

        let mcc_risk = self.mcc_risk.get(&req.merchant.mcc).copied().unwrap_or(0.5);

        let merchant_avg_amount = math::clamp_ratio(
            req.merchant.avg_amount as f32,
            self.normalization.max_merchant_avg_amount,
        );

        Ok([
            amount,
            installments,
            amount_vs_avg,
            hour_of_day,
            day_of_week,
            minutes_since_last_tx,
            km_from_last_tx,
            km_from_home,
            tx_count_24h,
            is_online,
            card_present,
            unknown_merchant,
            mcc_risk,
            merchant_avg_amount,
        ])
    }

    fn get_last_tx_data(
        &self,
        req: &FraudScoreRequest,
        requested_at: DateTime<Utc>,
    ) -> Result<(f32, f32), FraudEngineError> {
        let Some(last_transaction) = &req.last_transaction else {
            return Ok((-1.0, -1.0));
        };

        let last_timestamp = loader::parse_utc_timestamp(&last_transaction.timestamp)?;

        let elapsed_seconds = requested_at
            .signed_duration_since(last_timestamp)
            .num_seconds()
            .max(0) as f32;

        let time = math::clamp_ratio(elapsed_seconds / 60.0, self.normalization.max_minutes);

        let distance = math::clamp_ratio(
            last_transaction.km_from_current as f32,
            self.normalization.max_km,
        );

        Ok((time, distance))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::model::{
        Customer, FraudScoreRequest, LastTransaction, Merchant, Terminal, Transaction,
    };

    use super::*;

    fn engine() -> FraudEngine {
        FraudEngine::load(Path::new("./spec/resources")).expect("spec resources should load")
    }

    #[test]
    fn vectorizes_missing_last_transaction_with_sentinel_values() {
        let request = FraudScoreRequest {
            id: "tx-1329056812".to_owned(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".to_owned(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".to_owned(), "MERC-016".to_owned()],
            },
            merchant: Merchant {
                id: "MERC-016".to_owned(),
                mcc: "5411".to_owned(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        };

        let vector = engine()
            .vectorize(&request)
            .expect("vector should be produced");

        assert_eq!(vector[5], -1.0);
        assert_eq!(vector[6], -1.0);
        assert_eq!(vector[9], 0.0);
        assert_eq!(vector[10], 1.0);
        assert_eq!(vector[11], 0.0);
        assert!((vector[12] - 0.15).abs() < 0.0001);
    }

    #[test]
    fn vectorizes_known_fraud_shape_with_previous_transaction() {
        let request = FraudScoreRequest {
            id: "tx-1788243118".to_owned(),
            transaction: Transaction {
                amount: 4368.82,
                installments: 8,
                requested_at: "2026-03-17T02:04:06Z".to_owned(),
            },
            customer: Customer {
                avg_amount: 68.88,
                tx_count_24h: 18,
                known_merchants: vec![
                    "MERC-004".to_owned(),
                    "MERC-015".to_owned(),
                    "MERC-017".to_owned(),
                    "MERC-007".to_owned(),
                ],
            },
            merchant: Merchant {
                id: "MERC-062".to_owned(),
                mcc: "7801".to_owned(),
                avg_amount: 25.55,
            },
            terminal: Terminal {
                is_online: true,
                card_present: false,
                km_from_home: 881.61,
            },
            last_transaction: Some(LastTransaction {
                timestamp: "2026-03-17T01:58:06Z".to_owned(),
                km_from_current: 660.92,
            }),
        };

        let vector = engine()
            .vectorize(&request)
            .expect("vector should be produced");

        assert!((vector[0] - 0.4369).abs() < 0.001);
        assert!((vector[1] - 0.6667).abs() < 0.001);
        assert_eq!(vector[9], 1.0);
        assert_eq!(vector[10], 0.0);
        assert_eq!(vector[11], 1.0);
        assert!((vector[12] - 0.8).abs() < 0.001);
    }
}
