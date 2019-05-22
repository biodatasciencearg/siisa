---select top 2 * from dbo.RulePolicyExecutions
---obtener la tabla de pass y clientes del motor.
select * from sbrde.dbo.ClientUsers
order by clientid desc
--------

----me traigo todas las variables que tienen que ver con un Id en particular.
select  * from dbo.RulePolicyExecutionDumps
WHERE dbo.RulePolicyExecutionDumps.executionId = 1419591

---me traigo la data de politica de ejecucion del id en particular.
SELECT * from dbo.RulePolicyExecutions
WHERE dbo.RulePolicyExecutions.executionId = 1419591

---intento de hacer un PIVOT TABLE de modo tal que me ponga las variables como columnas.
DECLARE @DynamicPivotQuery AS NVARCHAR(MAX)
DECLARE @ColumnName AS NVARCHAR(MAX)
 
--Get distinct values of the PIVOT Column 
SELECT @ColumnName= ISNULL(@ColumnName + ',','') 
       + QUOTENAME(dbo.RulePolicyExecutionDumps.executionId)
FROM (SELECT DISTINCT dbo.RulePolicyExecutionDumps.executionId FROM dbo.RulePolicyExecutionDumps WHERE dbo.RulePolicyExecutionDumps.executionId = 1419591 ) AS executionId
 
--Prepare the PIVOT query using the dynamic 
SET @DynamicPivotQuery = 
  N'SELECT varName, ' + @ColumnName + '
    FROM dbo.RulePolicyExecutionDumps
    PIVOT(SUM(dbo.RulePolicyExecutionDumps.varValue) 
          FOR dbo.RulePolicyExecutionDumps.executionId IN (' + @ColumnName + ')) AS PVTTable'
--Execute the Dynamic Pivot Query
EXEC sp_executesql @DynamicPivotQuery

--SELECT top 5 * FROM dbo.RulePolicyExecutionDumps  WHERE (dbo.RulePolicyExecutionDumps.varName LIKE '*veraz*')

select top 15 dbo.RulePolicyExecutionDumps.executionId

SELECT * from dbo.RulePolicyExecutions
WHERE dbo.RulePolicyExecutions.executionId = 1715351

---me quedo con los primeros 10 datos del execution id del cliente y politica que me interesa evaluar.
select top 10 * from dbo.RulePolicyExecutions WHERE (dbo.RulePolicyExecutions.clientId=127)  AND (dbo.RulePolicyExecutions.policyId = 1)

---cuento la cantidad de runs que hay para la politica #1 del cliente127
select COUNT(*)  from dbo.RulePolicyExecutions WHERE (dbo.RulePolicyExecutions.clientId=127)  AND (dbo.RulePolicyExecutions.policyId = 1)

--Me quedo con las execs donde pueda ver lo que decidio
select * from dbo.RulePolicyExecutionDumps
where (dbo.RulePolicyExecutionDumps.executionId IN (select  dbo.RulePolicyExecutions.executionId from dbo.RulePolicyExecutions WHERE (dbo.RulePolicyExecutions.clientId=127)  AND (dbo.RulePolicyExecutions.policyId = 1))) AND dbo.RulePolicyExecutionDumps.varName='decisionResult'

select top 20 * from dbo.RulePolicyParameters




