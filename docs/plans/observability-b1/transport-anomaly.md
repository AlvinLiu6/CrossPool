# Transport Anomaly Evidence

The [C0 v7 shared contract](https://github.com/saitewasreset/CrossPool/blob/feat/observability_c0/docs/plans/unified-timeline-observability/shared-contract.md#anomalous-transport-evidence)
owns the normative payload and frozen vectors. Kind 150 and RawRequest bit 18
preserve invalid received request enums at actual Transport boundaries. Required
presence is 0x50007; only Result bit 15 may be added. Bytes 136–143 remain zero,
and presence bit 8 is forbidden. data0 names the original boundary, data1/data2
retain raw enum words and their exact invalid mask, and data3/data4 retain the
received layer and row count.

Keep actual local results and observed raw zero timestamps. Terminal-only
survival does not supply opening times, and an opening Event alone proves no
result publication. Legal enums with invalid geometry remain ordinary Events.
Full, abandoned, transfer-lost and unpublished evidence remain explicit losses
or unknown tails rather than invented successful facts. No legacy Record or
snapshot evidence report is reconstructed.
