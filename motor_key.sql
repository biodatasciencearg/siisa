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

 
--SELECT top 5 * FROM dbo.RulePolicyExecutionDumps  WHERE (dbo.RulePolicyExecutionDumps.varName LIKE '*veraz*')

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

select count(*)  from dbo.RulePolicyExecutions WHERE (dbo.RulePolicyExecutions.clientId=133)  AND (dbo.RulePolicyExecutions.policyId = 1)
select top 10 * from dbo.RulePolicyExecutions WHERE (dbo.RulePolicyExecutions.clientId=133)  AND (dbo.RulePolicyExecutions.policyId = 1)

select * from information_schema.columns
where TABLE_NAME='RulePolicyExecutionDumps'


      FOR JSON AUTO

select  * from dbo.RulePolicyExecutionDumps
WHERE dbo.RulePolicyExecutionDumps.executionId = 1419591
FOR JSON AUTO


SELECT   * from dbo.variablesVeraz


SELECT * from dbo.RulePolicyExecutions
WHERE dbo.RulePolicyExecutions.executionId = 1715351

SELECT *
FROM dbo.RulePolicyExecutions
WHERE dbo.RulePolicyExecutions.executionId < '20190526' AND dbo.RulePolicyExecutions.executionId > '20190525'


----query para traerme la data de dumps para un dado bloque de tiempo.
select  * from dbo.RulePolicyExecutionDumps
WHERE dbo.RulePolicyExecutionDumps.executionId  IN (
select dbo.RulePolicyExecutions.executionId
  from dbo.RulePolicyExecutions
 where dbo.RulePolicyExecutions.executionDate between '20190301 00:00:00' and '20190301 00:01:00')
 ----------------------------------------------------------------------------------------------------------

