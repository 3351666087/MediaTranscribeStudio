package com.mediatranscribestudio.pdf.contract;

import java.util.ArrayList;
import java.util.List;

public final class RepairQueue {
    public String schemaVersion = "1.0.0";
    public String documentId;
    public Integer round;
    public String status;
    public List<QualityReport.Repair> repairs = new ArrayList<>();
}
