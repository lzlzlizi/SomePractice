DROP TABLE IF EXISTS `appdata`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `appdata` (
  `word` varchar(45) NOT NULL,
  `level` int(11) DEFAULT NULL,
  `last_date` datetime DEFAULT NULL,
  PRIMARY KEY (`word`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;


DROP TABLE IF EXISTS `dicts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `dicts` (
  `word` varchar(50) NOT NULL,
  `sound` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `eng` varchar(500) DEFAULT NULL,
  `chinese` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  PRIMARY KEY (`word`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `diff`
--

DROP TABLE IF EXISTS `diff`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `diff` (
  `word` varchar(45) NOT NULL,
  `day` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `diff0`
--

DROP TABLE IF EXISTS `diff0`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `diff0` (
  `word` varchar(45) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;


DROP TABLE IF EXISTS `difficult_word`;
/*!50001 DROP VIEW IF EXISTS `difficult_word`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `difficult_word` AS SELECT 
 1 AS `word`,
 1 AS `remember`,
 1 AS `forget`,
 1 AS `next_level`,
 1 AS `difficult`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `gre_3000`
--

DROP TABLE IF EXISTS `gre_3000`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `gre_3000` (
  `word` varchar(50) NOT NULL,
  `def` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  PRIMARY KEY (`word`,`def`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;


DROP TABLE IF EXISTS `group5_30`;
/*!50001 DROP VIEW IF EXISTS `group5_30`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `group5_30` AS SELECT 
 1 AS `word`,
 1 AS `time`,
 1 AS `status_now`,
 1 AS `remembered`*/;
SET character_set_client = @saved_cs_client;

--
-- Temporary view structure for view `hard`
--

DROP TABLE IF EXISTS `hard`;
/*!50001 DROP VIEW IF EXISTS `hard`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `hard` AS SELECT 
 1 AS `word`,
 1 AS `forget`,
 1 AS `level`,
 1 AS `ratio`*/;
SET character_set_client = @saved_cs_client;

--
-- Temporary view structure for view `hard_today`
--

DROP TABLE IF EXISTS `hard_today`;
/*!50001 DROP VIEW IF EXISTS `hard_today`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `hard_today` AS SELECT 
 1 AS `word`,
 1 AS `forget`,
 1 AS `remember`,
 1 AS `hhd`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `mylog`
--

DROP TABLE IF EXISTS `mylog`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mylog` (
  `word` varchar(45) NOT NULL,
  `time` datetime NOT NULL,
  `status_now` int(2) DEFAULT NULL,
  `remembered` bit(1) DEFAULT NULL,
  PRIMARY KEY (`word`,`time`),
  KEY `da` (`time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

DROP TABLE IF EXISTS `questions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `questions` (
  `question` varchar(1000) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `selections` varchar(1000) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `answer` varchar(10) DEFAULT NULL,
  `words` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`question`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;



DROP TABLE IF EXISTS `remembering`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `remembering` (
  `word` varchar(45) NOT NULL,
  `status` int(11) DEFAULT NULL,
  PRIMARY KEY (`word`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tablename`
--

DROP TABLE IF EXISTS `tablename`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tablename` (
  `word` text,
  `status_now` bigint(20) DEFAULT NULL,
  `remembered` bigint(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;


DROP TABLE IF EXISTS `words_review`;
/*!50001 DROP VIEW IF EXISTS `words_review`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `words_review` AS SELECT 
 1 AS `word`,
 1 AS `level`,
 1 AS `last_date`,
 1 AS `from_now`*/;
SET character_set_client = @saved_cs_client;