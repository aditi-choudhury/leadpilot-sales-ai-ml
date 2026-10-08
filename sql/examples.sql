-- Latest prediction for each lead
SELECT l.lead_id,p.probability,p.priority,p.created_at FROM leads l JOIN predictions p ON p.id=(SELECT MAX(id) FROM predictions WHERE lead_id=l.lead_id) ORDER BY p.probability DESC LIMIT 20;
-- Lead volume by source
SELECT lead_source,COUNT(*) AS leads FROM leads GROUP BY lead_source ORDER BY leads DESC;
