SELECT 
        k.word AS word,
        COUNT(IF((k.remembered = 1), 1, NULL)) AS remember,
        COUNT(IF((k.remembered = 0), 1, NULL)) AS forget,
        COUNT(IF(((k.remembered = 1) AND (k.status_now > 0)),1,NULL)) AS n_next_level,
        (COUNT(IF((k.remembered = 0), 1, NULL)) / COUNT(IF(((k.remembered = 1) AND (k.status_now > 0)),1,NULL)))
             AS difficult
    FROM
        (SELECT 
            raw.word AS word,
                raw.status_now AS status_now,
                raw.remembered AS remembered,
                raw.time AS time
        FROM
            mylog raw
        WHERE (raw.time > (
                SELECT mylog.time FROM mylog
                WHERE ((mylog.word = raw.word)
                        AND (mylog.status_now = 1)
                        AND (mylog.remembered = 1))
                ORDER BY mylog.time DESC
                LIMIT 1  offset  4))) k
    GROUP BY k.word
    ORDER BY difficult DESC